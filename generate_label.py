import argparse
import torch
import os

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
from dassl.data.datasets.da import office_home
from datasets.office31_pseudo_xshots import Office31Pseudo
from datasets.visda_pseudo import Visda2017Pseudo
from datasets.officehome_pseudo_xshots import OfficeHomePseudo_xshots
from datasets.mini_domainet_pseudo import MiniDomainNetPseudo

from trainers.source_adapter import Source_adapter
from trainers.zsclip import ZeroshotCLIP


import numpy as np
import random

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seeds(0)

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    
    
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head
    if args.gpu:
        cfg.GPU_ID = args.gpu

    if args.disable_mapper:
        cfg.IS_USER_DOMAIN_MAPPER = False
    
    if args.cls_rate:
        cfg.cls_rate = args.cls_rate

    else:
        cfg.GPU_ID = 0


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    # cfg.CLASS_NUM = 65
    # cfg.DATASET.NUM_SHOTS = 4
    cfg.MAPPER_ALPHA = 0.5
    cfg.BETA = 0 # 原型对齐损失中的一个参数 设置为0即可
    cfg.IS_USE_MAPPER = False
    
    cfg.IS_USER_DOMAIN_MAPPER = True
    
    cfg.max_batch_one_epoch = 65

    
    #控制损失项的参数，0表示不使用该损失项
    cfg.IM_WEIGHT = 1   ####需要消融的损失
    cfg.cls_rate = 0.2    ####需要消融的损失

    cfg.U_NUM_SHOTS = -1
    cfg.eval_only = args.eval_only
    cfg.is_resample_target = False
    

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    source_marker = cfg.DATASET.SOURCE_DOMAINS[0][0].upper()  # 获取 source 元组第一个元素的首字母并转换为大写
    target_marker = cfg.DATASET.TARGET_DOMAINS[0][0].upper()  # 获取 target 元组第一个元素的首字母并转换为大写
    
    # 构造数据集标记信息
    dataset_marker = f"{source_marker}2{target_marker}"
    
    # 更新输出目录
    cfg.OUTPUT_DIR = os.path.join(args.output_dir, dataset_marker)
    
    # 确保目录存在
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer = Source_adapter(cfg)
        trainer.build_model()
        trainer.generate_pseudo_labels("/home/CDBN/datasets/officehome/zeroshotrefer/pseudo_truth_product.json")
        # trainer.test()
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/home/Datasets", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="/home/CDBN/output/officehome/vit", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )

    parser.add_argument(
        "--config-file", type=str, default="/home/CDBN/configs/trainers/CoOp/vit_b16.yaml", help="path to config file"
    )
    parser.add_argument(
        "--disable-mapper", action="store_true", help="disable mapper"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="/home/CDBN/configs/datasets/office_home.yaml",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="ZeroshotCLIP", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument("--cls-rate", type=float, default=0.2, help="parameter for cls loss")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)