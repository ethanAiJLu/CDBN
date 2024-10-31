import torch
import torch.nn as nn
import numpy as np

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.model import convert_weights
from clip.clip import _transform

from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT
from tqdm import tqdm
import json
import os

from datasets.data_manage_u import build_data_loader

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "OfficeHomePseudo_xshots" : "a photo of a {}",
    "MiniDomainNetPseudo" : "a photo of a {}",
    "Visda2017Pseudo" : "a photo of a {}.",
    "Office31Pseudo" : "a photo of a {}."
}

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

@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg.MODEL.BACKBONE.NAME)
        clip_model.to(self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " "),cfg.DATASET.TARGET_DOMAINS[0].lower()) for c in classnames]
        # prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model
        self.target_data = self.dm.dataset.train_u
        self.dataset_size = len(self.target_data)
        img_process = _transform(224)
        self.custom_tfm_train = [img_process]

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits
    
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
            output= self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)            

        return list(results.values())[0]
    
    @torch.no_grad()
    def generate_pseudo_labels(self, save_path, split=None):
        """Generate pseudo labels and save to JSON."""
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

        results = []

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label, img_path, domain, classnames = self.parse_batch_pseudo_label(batch)
            logits = self.model_inference(input)
            probs = logits.softmax(dim=-1)
            confs, preds = probs.max(dim=-1)

            for i in range(input.size(0)):
                result = {
                    "img_path": img_path[i],
                    "domain": domain[i],
                    "classname": classnames[preds[i]].lower(),
                    "confidence": confs[i].item(),
                    "label": preds[i].item(),
                    "ground_truth": label[i].item(),
                }
                results.append(result)
        
        for result in results:
            for key, value in result.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value.item()

        with open(save_path, "w", encoding="utf8") as file:
            json.dump({"datas": results}, file, ensure_ascii=False, indent=4)
    
    def parse_batch_pseudo_label(self, batch):
        input = batch["img"]
        label = batch["label"]
        img_path = batch["impath"]
        domain = batch["domain"]
        classnames = self.dm.dataset.classnames

        input = input.to(self.device)
        label = label.to(self.device)

        img_path = [p for p in img_path]
        domain = [d for d in domain]

        return input, label, img_path, domain, classnames

