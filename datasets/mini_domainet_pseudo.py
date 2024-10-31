import os.path as osp

from dassl.utils import listdir_nohidden
from dassl.data.datasets.build import DATASET_REGISTRY
from dassl.data.datasets.base_dataset import Datum, DatasetBase
import random

import json
from dassl.utils import check_isfile

class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath="", label=0, domain=0, classname="",confidence=0,ground_truth=0):
        assert isinstance(impath, str)
        assert check_isfile(impath)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname
        self._confidence = confidence
        self._ground_truth = ground_truth

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname
    
    @property
    def confidence(self):
        return self._confidence
    
    @property
    def ground_truth(self):
        return self._ground_truth
    
    def to_dict(self):
        return {
            'impath': self.impath,
            'label': self.label,
            'domain': self.domain,
            'classname': self.classname,
            'confidence': self.confidence,
            'ground_truth': self.ground_truth
        }
    


@DATASET_REGISTRY.register()
class MiniDomainNetPseudo(DatasetBase):
    """A subset of DomainNet.

    Reference:
        - Peng et al. Moment Matching for Multi-Source Domain
        Adaptation. ICCV 2019.
        - Zhou et al. Domain Adaptive Ensemble Learning.
    """

    dataset_dir = "domainnet"
    # dataset_dir = "DomainNet"
    domains = ["clipart", "painting", "real", "sketch"]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.split_dir = osp.join(self.dataset_dir, "splits_mini")
        self.shots = cfg.xshots

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train_x = self._read_data(cfg.DATASET.SOURCE_DOMAINS, split="train")
        train_u = self._read_data(cfg.DATASET.TARGET_DOMAINS, split="train")
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS, split="test")
        self.support_u = self._read_data_support_target(cfg.DATASET.TARGET_DOMAINS)
        
        if cfg.DATASET.NUM_SHOTS > 0:
            train_x = self.sample_num_shots(train_x,cfg.DATASET.NUM_SHOTS)
        self.target_portion = self.portion(train_u)

        super().__init__(train_x=train_x, train_u=train_u, test=test)
        
        
    def sample_num_shots(self,items,num_shots):
        label_2_samples = {}
        for item in items:
            item_label = item.label
            if item_label in label_2_samples:
                label_2_samples[item_label].append(item)
            else:
                label_2_samples[item_label] = [item]
        targets = []
        for label, items in label_2_samples.items():
            item_target = []
            if num_shots > len(items):
                n = int(num_shots/len(items))
                for i in range(n):
                    item = random.sample(items, len(items))
                    item_target.extend(item)
                item = random.sample(items,num_shots - len(item_target))
                item_target.extend(item)
            else:
                item = random.sample(items, num_shots)
                item_target.extend(item)
            targets.extend(item_target)
        return targets
    
    def portion(self,items):
        sum = len(items)
        item_dict = {}
        for item in items:
            item_label = item.label
            if item_label in item_dict:
                item_dict[item_label]+=1
            else:
                item_dict[item_label] = 1
        por = [item_dict[key]/sum for key in sorted(item_dict.keys())]
        return por
    

    def _read_data(self, input_domains, split="train"):
        items = []

        for domain, dname in enumerate(input_domains):
            filename = dname + "_" + split + ".txt"
            split_file = osp.join(self.split_dir, filename)

            with open(split_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    impath, label = line.split(" ")
                    classname = impath.split("/")[1]
                    impath = osp.join(self.dataset_dir, impath)
                    label = int(label)
                    item = Datum(
                        impath=impath,
                        label=label,
                        domain=domain,
                        classname=classname
                    )
                    items.append(item)

        return items
    
    def _read_data_support_target(self, input_domains):
        items = []
        path = "/home/CDBN/datasets/domainnet/zeroshotrefer"
        data_by_class = {}

        for domain, dname in enumerate(input_domains):
            domain_index = self.domains.index(dname)
            json_path = osp.join(path, f'pseudo_truth_{dname}.json')
            print("read data from:", str(json_path))

            # Read JSON file
            with open(json_path, 'r') as f:
                data = json.load(f)

            for entry in data['datas']:
                impath = entry['img_path']
                label = entry['label']
                confidence = entry['confidence']
                classname = entry['classname']
                ground_truth = entry['ground_truth']

                item = Datum(
                    impath=impath,
                    label=label,
                    domain=domain_index,
                    classname=classname,
                    confidence=confidence,
                    ground_truth=ground_truth
                )

                if classname not in data_by_class:
                    data_by_class[classname] = []

                data_by_class[classname].append(item)
        topn = self.shots
        # For each class, sort by confidence and select top n
        for classname, items_list in data_by_class.items():
            sorted_items = sorted(items_list, key=lambda x: x.confidence, reverse=True)
            top_items = sorted_items[:topn]
            
            if len(top_items) < topn:
                num_to_add = topn - len(top_items)
                for i in range(num_to_add): 
                    top_items.append(sorted_items[0])
                # top_items.extend(random.sample(top_items, topn - len(top_items)))
            items.extend(top_items)
             
        items.sort(key=lambda x: x.label)
       
        return items
    
