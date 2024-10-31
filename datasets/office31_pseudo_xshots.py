import os.path as osp
from collections import defaultdict

from dassl.utils import listdir_nohidden
from dassl.data.datasets.build import DATASET_REGISTRY
from dassl.data.datasets.base_dataset import DatasetBase
import random
import json
import os
from my_utils import predictor
from tqdm import  tqdm
from sklearn.metrics._classification import classification_report
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
class Office31Pseudo(DatasetBase):
    
    dataset_dir = "office31"
    domains = ["amazon", "dslr", "webcam"]
    
    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.domains = ["amazon", "dslr", "webcam"]

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )
        self.shots = cfg.xshots
        train_x = self._read_data(cfg.DATASET.SOURCE_DOMAINS)
        train_u = self._read_data(cfg.DATASET.TARGET_DOMAINS)
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS)
        self.support_u, self.empty_categories, self.empty_indices = self._read_data_support_target(cfg.DATASET.TARGET_DOMAINS)
        # print("-----test-------")
        # print(self.empty_categories)
        # print(self.empty_indices)
        self.train_x_test = train_x
        self.target_portion = self.portion(train_u)
        
 
        if cfg.DATASET.NUM_SHOTS > 0:
            train_x = self.sample_num_shots(train_x,cfg.DATASET.NUM_SHOTS)
            self.save_num_shots_to_local(train_x,cfg.OUTPUT_DIR)

        super().__init__(train_x=train_x, train_u=train_u, test=test)
        
        
    
    def save_num_shots_to_local(self, datas, save_dir):
        data_dict = {"datas":[]}
        for item in datas:
            data_dict["datas"].append(
                {"img_path":item.impath,
                        "label":item.label,
                        "domain":item.domain,
                        "classname":item.classname.lower()
                 }
            )
        with open(os.path.join(save_dir,"num_shots_datas.json"),"w",encoding="utf8") as file:
            json.dump(data_dict,file)
            
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
    
    def _read_data(self, input_domains):
        items = []

        for domain, dname in enumerate(input_domains):
            domain_index = self.domains.index(dname)
            domain_dir = osp.join(self.dataset_dir, dname)
            class_names = listdir_nohidden(domain_dir)
            class_names.sort()

            for label, class_name in enumerate(class_names):
                class_path = osp.join(domain_dir, class_name)
                imnames = listdir_nohidden(class_path)

                for imname in imnames:
                    impath = osp.join(class_path, imname)
                    item = Datum(
                        impath=impath,
                        label=label,
                        domain=domain_index,
                        classname=class_name.lower(),
                    )
                    items.append(item)

        return items
        

    def _read_data_support_target(self, input_domains):
        items = []
        empty_indices = []
        empty_categories = []
        path = "/home/CDBN/datasets/office31/zeroshotrefer"
        data_by_class = {}
        categroy_to_index = {}
        

        for domain, dname in enumerate(input_domains):
            domain_index = self.domains.index(dname)
            domain_index = self.domains.index(dname)
            domain_dir = osp.join(self.dataset_dir, dname)
            class_names = listdir_nohidden(domain_dir)
            class_names.sort()
            category_samples = {category:[] for category in class_names}
            
            for idx, classname in enumerate(class_names):
                categroy_to_index[classname] = idx
            
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

                # if classname not in data_by_class:
                #     data_by_class[classname] = []

                # data_by_class[classname].append(item)
                category_samples[classname].append(item)
        topn = self.shots
        # For each class, sort by confidence and select top n
        # for classname, items_list in data_by_class.items():
        for classname, items_list in category_samples.items():
            sorted_items = sorted(items_list, key=lambda x: x.confidence, reverse=True)
            top_items = sorted_items[:topn]
            if len(top_items) == 0:
                empty_categories.append(classname)
            if len(top_items) < topn and len(top_items) > 0:
                num_to_add = topn - len(top_items)
                for i in range(num_to_add): 
                    top_items.append(sorted_items[0])
                # top_items.extend(random.sample(top_items, topn - len(top_items)))
            items.extend(top_items)
        items.sort(key=lambda x: x.label)
        
        empty_indices = [categroy_to_index[classname] for classname in empty_categories if classname in categroy_to_index]
        
        return items, empty_categories, empty_indices
    