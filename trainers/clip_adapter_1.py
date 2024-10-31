import os.path as osp
import sys
import copy

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms

from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from typing import Optional
from clip import clip
from clip.clip import _transform
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from PIL import Image
from tqdm import tqdm
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
import time
import datetime
import math
from scipy.stats import entropy
import numpy as np
from dassl.data import DataManager
from dassl.data.transforms import build_transform

sys.path.append('../')
from datasets.data_manage_u import DataManagerX


_tokenizer = _Tokenizer()

def get_entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def compute_im_loss(logits):
    softmax_out = nn.Softmax(dim=1)(logits)
    entropy_loss = torch.mean(get_entropy(softmax_out))
    msoftmax = softmax_out.mean(dim=0)
    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-6))
    im_loss = entropy_loss - gentropy_loss
    # return im_loss
    return torch.log(1 + torch.exp(im_loss)), im_loss


class CA_Adapter(nn.Module):
    def __init__(self, dim=512, num_heads=1, key_vector=None):
        super().__init__()
        self.key = nn.Parameter(key_vector) if key_vector is not None else None  # to be optimized
        self.dim = dim
        self.num_heads = num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.alpha_proj = nn.Linear(dim, 1, bias=True)
        self._reset_parameters()
        
        self.q_proj.weight.data = self.q_proj.weight.data.to(torch.float16)
        self.k_proj.weight.data = self.k_proj.weight.data.to(torch.float16)
        self.alpha_proj.weight.data = self.alpha_proj.weight.data.to(torch.float16)
        if self.alpha_proj.bias is not None:
            self.alpha_proj.bias.data = self.alpha_proj.bias.data.to(torch.float16)
        else:
            print("Warning: alpha_proj.bias is None")
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.alpha_proj.weight)
        nn.init.constant_(self.alpha_proj.bias, 1)

    def forward(self, query):
        key = self.key
        value = key
        B, K, C = key.shape 
        res = query
        query = query.reshape(B, 1, C) 
        key = torch.cat([query, key], dim=1) 
        value = torch.cat([query, value], dim=1) 
        query = self.q_proj(query).reshape(B, self.num_heads, C)
        key = self.k_proj(key)

        query = query.reshape(B, self.num_heads, 1, -1).permute(0, 2, 1, 3)
        key = key.reshape(B, K + 1, 1, -1).permute(0, 2, 1, 3)
        value = value.reshape(B, K + 1, 1, -1).permute(0, 2, 1, 3)

        attn_weight = (query @ key.transpose(-1, -2) / torch.sqrt(torch.tensor(self.dim, dtype=torch.float16,device = query.device))).softmax(-1)
        attn = attn_weight @ value

        alpha = torch.sigmoid(self.alpha_proj(res).reshape(B, -1, 1, 1))
        attn = (alpha * attn).squeeze()

        attn = res + attn
        attn = F.normalize(attn, p=2, dim=-1)
        return attn

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.cfg = cfg
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
               
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        if cfg.freeze_prompt:
            self.ctx.requires_grad = False
            
        self.eval_only = cfg.eval_only
        self.random_init = cfg.random_init

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        
        self.name_lens = name_lens
        self.min_len = min(self.name_lens)
        if self.min_len > 1:
            print("origin len is:", name_lens)
            classnames = self.revise_classnames(classnames, name_lens, self.min_len)
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            print("later len is:", name_lens)
        
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        
        self.n_cls = n_cls
        print(f"Number of classes: {n_cls}")
        self.n_ctx = n_ctx

        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
        
        self._init_suffix_dict(classnames, clip_model, dtype)
        self._get_token_classes(dtype)
        if cfg.freeze_classes:
                self.token_classes.requires_grad = False
        
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.min_len + n_ctx:, :])  # EOS
        self.register_buffer("token_suffix_test", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

    def construct_prompts(self, ctx, prefix, suffix):
        
        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        else:
            raise ValueError
        
        return prompts
    
    def construct_prompts_v2(self, ctx, prefix, classes, suffix):
        
        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    classes, # (dim0, 1, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        else:
            raise ValueError
        
        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        
        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            classes = self.token_classes
            prompts = self.construct_prompts_v2(ctx, prefix, classes, suffix)
           
            

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts
       
    def forward_test(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix_test
    
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts
    
    def _init_suffix_dict(self, classnames, clip_model, dtype):
            
        self.suffix_classes = {}
        for name in classnames:
            self.suffix_classes[name] = clip_model.token_embedding(clip.tokenize(name)).type(dtype)
    
    def _get_token_classes(self, dtype):
        
        if not self.eval_only:
            self.token_classes_all = torch.cat([self.suffix_classes[name] for name in self.suffix_classes]).type(dtype)            
            self.token_classes = self.token_classes_all[:, 1:self.min_len+1, :]
            if self.random_init:
                nn.init.normal_(self.token_classes, std=0.02)
            self.token_classes = nn.Parameter(self.token_classes)
            
            self.fix_token = copy.deepcopy(self.token_classes)
            self.fix_token.requires_grad = False
        else:
            pass 

    def revise_classnames(self, classnames, name_lens, min_len):
        if min(name_lens) < min_len:
            for i in range(len(classnames)):
                if name_lens[i] < min_len:
                    classnames[i] = ("<|startoftext|> "*(min_len - name_lens[i])) + classnames[i]
        return classnames



def load_clip_to_cpu(NAME):
    backbone_name = NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model



class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        features_dim = clip_model.visual.output_dim
        self.cfg = cfg
        self.device = device
        train_tranform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        img_process = _transform(224)
        custom_tfm_train = [img_process]
        choices = cfg.TRAINER.FIXMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManagerX(self.cfg, custom_tfm_train=custom_tfm_train, custom_tfm_test = img_process)
        self.train_loader_support_u = dm.train_loader_support_u
        if hasattr(dm, 'empty_indices'):
            self.empty_indices = dm.empty_indices
            self.empty_categories = dm.empty_categories
        self.num_classes = dm.num_classes
        self.dtype = clip_model.dtype
        self.class_num = len(classnames)

        temp = 'a photo of a {}.'
        prompts = [temp.format(c.replace("_", " "),cfg.DATASET.TARGET_DOMAINS[0].lower()) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)
        clip_model.to(self.device)
        
        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features.to(self.device)
        self.clip_model = clip_model 
        query = self.text_features
        cache_keys = self.build_support_model(cfg, clip_model)
        cache_keys_reshape = cache_keys.T.reshape(query.shape[0], -1, query.shape[1])
        self.cache_keys = cache_keys_reshape
        print("shape of cache key:",cache_keys_reshape.shape)
        self.CA_adapter = CA_Adapter(dim=cache_keys.shape[0], key_vector = cache_keys_reshape).to(self.device)
        

    def forward(self, image, is_source):
        # [batchsize,dim]
        clip_image_features = self.image_encoder(image.type(self.dtype))
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
            
        text_features = self.text_encoder(prompts, tokenized_prompts)
          
        image_features = clip_image_features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # 已经归一化过了
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        # text_feature 为 k dim
        return logits,image_features,text_features,clip_image_features   
    
    def generate_clip_text_feature(self, category, clip_model):
        temp = "a photo of a {}."
        prompt = temp.format(category.replace("_", " ").lower())
    
        prompt = clip.tokenize(prompt).to(self.device)
    
        with torch.no_grad():
            text_feature = clip_model.encode_text(prompt)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
    
        return text_feature # 1 dim
    
    def build_support_model(self, cfg, clip_model):
        if cfg['load_cache'] == False:
            cache_keys = []
            with torch.no_grad():
                # Data augmentation for the cache model
                for augment_idx in range(cfg['augment_epoch']):
                    train_features = []
                    print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                    for batch_u in tqdm(self.train_loader_support_u):
                        images = batch_u["img"]
                        images = images.to(self.device)
                        image_features = clip_model.encode_image(images)
                        train_features.append(image_features)
                    cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

            cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
            cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
            if hasattr(self,'empty_categories'):
                empty_categories = self.empty_categories
                empty_indices = self.empty_indices
                text_features = []
            
                for classname in empty_categories:
                    text_feature = self.generate_clip_text_feature(classname, self.clip_model)
                    text_features.append(text_feature)
                text_features = torch.cat(text_features, dim=0)
                print("text_features+shape:", text_features.shape)
                num_classes = self.num_classes
                print("cachekeyshape:",cache_keys.shape)
                final_cache_keys = torch.zeros(num_classes * cfg.xshots, cache_keys.size(1), dtype=torch.float16, device = self.device)
                cache_idx = 0
                text_feature_dict = {idx: text_features[i] for i, idx in enumerate(empty_indices)}
                for i in range(num_classes):
                    if i in text_feature_dict:
                        for j in range(4):
                            final_cache_keys[i+j] = text_feature_dict[i]
                    else:
                        final_cache_keys[i] = cache_keys[i]
                    cache_idx += 1
                cache_keys = final_cache_keys
                print("cachekeys type :",cache_keys.dtype)
            cache_keys = cache_keys.permute(1, 0)
                
                        
            torch.save(cache_keys, cfg['cache_dir'] +str(cfg.DATASET.TARGET_DOMAINS[0]) + '/keys_' + str(cfg.xshots) + "shots.pt")
        else:
            cache_keys = torch.load(cfg['cache_dir'] +str(cfg.DATASET.TARGET_DOMAINS[0]) +'/keys_' + str(cfg.xshots) + "shots.pt", map_location=self.device)

        return cache_keys

@TRAINER_REGISTRY.register()
class Clip_adapter_office31(TrainerXU):
    
    def __init__(self, cfg):
        super().__init__(cfg)
       
        
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]
        assert len(cfg.TRAINER.FIXMATCH.STRONG_TRANSFORMS) > 0
        self.weight_u = cfg.TRAINER.FIXMATCH.WEIGHT_U
        self.conf_thre = cfg.TRAINER.FIXMATCH.CONF_THRE
        
    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        cfg = self.cfg 
        img_process = _transform(224)
        custom_tfm_train = [img_process]
        choices = cfg.TRAINER.FIXMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManagerX(self.cfg, custom_tfm_train=custom_tfm_train, custom_tfm_test = img_process)
        
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.train_loader_support_u = dm.train_loader_support_u_train
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm
        
    
    def get_text_features(self,clip_model):
        temp = "a photo of a {}."
        prompts = [temp.format(c.replace("_", " ").lower()) for c in self.classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)
        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # k dim
        return text_features

    def build_model(self):
        cfg = self.cfg
        self.alpha = cfg.parameter_alpha
        classnames = self.dm.dataset.classnames
        self.classnames = classnames
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda:{0}".format(cfg.GPU_ID))
        else:
            self.device = torch.device("cpu")
        

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg.MODEL.BACKBONE.NAME)
        self.best_result = 0
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
        self.clip_model = clip_model
        self.dtype = clip_model.dtype

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, self.device)
        self.img_process = _transform(clip_model.visual.input_resolution)

        self.cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)

        print("Turning off gradients in both the image and the text encoder")
        
        for name, param in self.model.named_parameters():
            if not (("CA_adapter" in name) or ("prompt_learner" in name) ):
                param.requires_grad_(False)
        

        source_marker = cfg.DATASET.SOURCE_DOMAINS[0][0].upper()  
        target_marker = cfg.DATASET.TARGET_DOMAINS[0][0].upper()  
        dataset_marker = f"{source_marker}2{target_marker}"
        model_name = "prompt_learner/model-last.pth.tar"
        weight_load_path = osp.join(cfg.MODEL.INIT_WEIGHTS, dataset_marker,model_name)
        
        if cfg.MODEL.INIT_WEIGHTS:
            print("loading prompt learner's source pretrained parameters..")
            load_pretrained_weights(self.model.prompt_learner, weight_load_path)
            

        self.model.to(self.device)
        self.clip_model.to(self.device)
        
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        
        
        self.adapter_optim = torch.optim.AdamW(self.model.CA_adapter.parameters(), lr=cfg['adapter_lr'], eps=1e-4)
        self.adapter_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.adapter_optim, cfg.OPTIM.MAX_EPOCH * cfg.max_batch_one_epoch)
        self.register_model("CA_adapter",self.model.CA_adapter, self.adapter_optim, self.adapter_scheduler)
        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None
        
        temp = "a photo of a "
        
        prompts = [temp.format(c.replace("_", " ").lower()) for c in self.classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        clip_model = clip_model.to(self.device)
        
        tokenized_prompts =  prompts.to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)
            
        self.embedding = embedding
        n_ctx = len(temp.split(" "))
        self.token_prefix = self.embedding[:, :1, :]  # SOS
        self.token_suffix = self.embedding[:, 1 + 1 + n_ctx:, :]  # EOS
        
        prefix = self.token_prefix
        suffix = self.token_suffix
        
        ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        if ctx_vectors.dim() == 2:
            ctx_vectors = ctx_vectors.unsqueeze(0).expand(len(self.classnames), -1, -1)
        
        token_classes = self.model.prompt_learner.token_classes
        
        
        newprompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx_vectors,     # (n_cls, n_ctx, dim)
                    token_classes , # (dim0, 1, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        newprompts = newprompts.to(dtype=torch.long)
        self.tokenized_prompts = self.model.prompt_learner.tokenized_prompts

        
        with torch.no_grad():
            self.newprompts = self.model.text_encoder(newprompts,tokenized_prompts)
            self.newprompts = self.newprompts / self.newprompts.norm(dim=-1, keepdim=True)
        
        self.model.prompt_learner.token_classes.to(self.device)
        
        
    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {
            "acc_thre": acc_thre,
            "acc_raw": acc_raw,
            "keep_rate": keep_rate
        }
        return output


    def forward_backward(self, batch_x, batch_u, batch_support_u):
        query = self.newprompts
        image_u, image_u2, label_u , u_size = self.parse_batch_train(batch_u)
        image_support_u, pseudo_label_support, support_u_size, u_confidence, ground_truth_u= self.parse_batch_train_support_u(batch_support_u) 
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                logit_u,image_features_u,text_features,clip_image_features_u = self.model(image_u,False)
                cls_loss =0.1 #  F.cross_entropy(logit_s, label_x)
                im_loss, _= compute_im_loss(logit_u)    
                loss = cls_loss * self.cfg.cls_rate + im_loss * self.cfg.IM_WEIGHT 

                
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            logit_u,image_features_u,text_features1,clip_image_features_u = self.model(image_u,False)
            logit_u2,image_features_u2,text_features2,clip_image_features_u2 = self.model(image_u2,False)
            logits_support, image_features_support, text_features_support, clip_image_features_support = self.model(image_support_u,False)
            
            output_u = F.softmax(logit_u, 1)
            max_prob, label_u_pred = output_u.max(1)
            mask_u = (max_prob >= self.conf_thre).float()
            y_u_pred_stats = self.assess_y_pred_quality(
                        label_u_pred, label_u, mask_u)
        
            weights = self.model.CA_adapter(query) # k dim
            tip_logits = 100. * image_features_u @ weights.T 
            
            logits_new = self.alpha * tip_logits + (1-self.alpha) * logit_u 
            
            loss_u = F.cross_entropy(logit_u2, label_u_pred, reduction="none")         
            loss_u = (loss_u * mask_u).mean()
            cls_loss = F.cross_entropy(logits_support, pseudo_label_support)
            im_loss, im_loss_minus = compute_im_loss(logits_new)
            
            loss = loss_u * self.weight_u + self.cfg.IM_MINUS * im_loss_minus + cls_loss * self.cfg.cls_rate
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "cls_loss": cls_loss.item(),
            "im_loss": im_loss.item(),
            "loss_u": loss_u.item(),
            "y_u_pred_acc_raw": y_u_pred_stats["acc_raw"],
            "y_u_pred_acc_thre": y_u_pred_stats["acc_thre"],
            "y_u_pred_keep": y_u_pred_stats["keep_rate"],
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
    

    def parse_batch_train(self, batch_x):
        input_x = batch_x["img"]
        input_x2 = batch_x["img2"]
        label_x = batch_x["label"]

        input_x = input_x.to(self.device)
        input_x2 = input_x2.to(self.device)
        label_x = label_x.to(self.device)

        return input_x, input_x2, label_x, input_x.shape[0]
    
    def parse_batch_train_support_u(self, batch_x):
        input_x = batch_x["img"]
        label_x = batch_x["label"]
        confidence = batch_x["confidence"]
        ground_truth = batch_x["ground_truth"]

        input_x = input_x.to(self.device)
        label_x = label_x.to(self.device)
        confidence = confidence.to(self.device)

        return input_x, label_x, input_x.shape[0], confidence, ground_truth
    
    def parse_batch_test(self, batch_x):
        input_x = batch_x["img"]
        label_x = batch_x["label"]

        input_x = input_x.to(self.device)
        label_x = label_x.to(self.device)

        return input_x, label_x

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
            
    def after_epoch(self):

        curr_result = self.test(split="test")
        is_best = curr_result > self.best_result
        if is_best:
            self.best_result = curr_result
            self.save_model(self.epoch,
                            self.output_dir,
                            model_name="model-best.pth.tar")

        self.set_model_mode("train")
    
        self.save_model(self.epoch, self.output_dir,model_name="model-last.pth.tar")
        
    def model_inference(self, input):
        return self.model(input,False)
        
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")
        
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label  = self.parse_batch_test(batch)
            output,img_features,text_features,clip_image_features = self.model_inference(input)
            weights = self.model.CA_adapter(text_features)
            tip_logits = 100. * img_features @ weights.T
            logits_new = self.alpha * tip_logits + output * (1-self.alpha)
          
            self.evaluator.process(logits_new, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
    
            
    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loader_x = len(self.train_loader_x) 
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError
        if self.cfg.max_batch_one_epoch > 0:
            self.num_batches = max(self.num_batches, self.cfg.max_batch_one_epoch)
       

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)
        train_loader_support_u_iter = iter(self.train_loader_support_u)
       

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)
            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)
            try:
                batch_support_u = next(train_loader_support_u_iter)
            except StopIteration:
                train_loader_support_u_iter = iter(self.train_loader_support_u)
                batch_support_u = next(train_loader_support_u_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u, batch_support_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()
    
    def cls_acc(self, output, target, topk=1):
        output = output.to(self.device)
        target = target.to(self.device)
        pred = output.topk(topk, 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        acc = 100 * acc / target.shape[0]

        return acc
    
    def pre_load_features(self, cfg, split, clip_model, loader, norm=True):
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.to(self.device), target.to(self.device)
                image_features = clip_model.encode_image(images)
                if norm:
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)  
        return features, labels
