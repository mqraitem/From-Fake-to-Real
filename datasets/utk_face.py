import os
import numpy as np
import pickle
import sys
from collections import namedtuple
from utils import *
from datasets.dataset import MainDataset

CSV = namedtuple("CSV", ["header", "index", "data"])

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


class UtkFace(MainDataset):
    def __init__(self, root_dir, split='train', transform=None, opt=None):
        super(UtkFace, self).__init__(root_dir, split, transform, opt)

        self.GROUPS = [(0, 0), (0, 1), (1, 0), (1, 1)]

        self.split_idx = { 
            'train': 0,
            'valid': 1,
            'test': 2
        }

        self.split_ratio = {
            'train': [0.0, 0.8],
            'valid': [0.8, 0.9],
            'test':  [0.9, 1.0],
        }

        self.target_attr = "gender"
        self.bias_attr = "age"

        self.class_names = ["Male", "Female"]
        self.bias_names = ["old", "young"]

        self.targets_values = range(2)
        self.biases_values = range(2)

        self.gen_dir = f"./generated_images/utk_face"        
    
        if os.path.exists(f"data/utk_face_{split}_{self.opt.minority_to_keep}.pkl"):
            data = pickle.load(open(f"data/utk_face_{split}_{self.opt.minority_to_keep}.pkl", "rb"))
            self.filenames = data['filenames']
            self.targets = data['targets']
            self.biases = data['biases']
            self.real_gen = data['real_gen']

        else: 
        
            self.load_filenames()
            self.load_data()
            self.fix_filenames()
            self.set_group_counts()

            if split == "train":
                self.make_harder(False)
                self.set_group_counts()

            if split == 'valid':
                self.make_harder(True)
                self.set_group_counts()
            # save filenames targets biases into a pickle file
            data = {
                'filenames': self.filenames,
                'targets': self.targets,
                'biases': self.biases,
                'real_gen': self.real_gen
            }
            pickle.dump(data, open(f"data/utk_face_{split}_{self.opt.minority_to_keep}.pkl", "wb"))

        self.set_group_counts()
        self.create_real_gen_weights()

    def get_class_from_filename(self, filenames, cls_idx):
        return np.array([int(fname.split('_')[cls_idx]) if len(fname.split('_')) == 4 else 10 for fname in filenames])

    def load_filenames(self):
        self.filenames = os.listdir(self.root_dir)
        start_idx = int(len(self.filenames) * self.split_ratio[self.split][0])
        end_idx = int(len(self.filenames) * self.split_ratio[self.split][1])
        
        self.filenames = self.filenames[start_idx:end_idx]

    def get_gen_dir(self, target_value, bias_value):
        return f"{self.class_names[target_value]}/{self.bias_names[bias_value]}" 

    def load_data(self):

        attr_dict = {
            'age': (0, lambda x: (x >= 20) & (x <= 50), lambda x: x <= 5,),
            'gender': (1, lambda x: x == 0, lambda x: x == 1),
            'race': (2, lambda x: x == 0, lambda x: x != 0),
        }

        assert self.target_attr in attr_dict.keys()
        target_cls_idx, *target_filters = attr_dict[self.target_attr]
        bias_cls_idx, *bias_filters = attr_dict[self.bias_attr]

        target_classes = self.get_class_from_filename(self.filenames, target_cls_idx)
        bias_classes = self.get_class_from_filename(self.filenames, bias_cls_idx)

        filenames = [] 
        targets = [] 
        biases = [] 
        for target_value in self.targets_values:
            for bias_value in self.biases_values:
                idx = np.where((target_filters[target_value](target_classes)) & (bias_filters[bias_value](bias_classes)))[0]

                filenames.extend([self.filenames[i] for i in idx])
                targets.extend([target_value] * len(idx))
                biases.extend([bias_value] * len(idx))
        
        self.filenames = filenames
        self.targets = np.array(targets)
        self.biases = np.array(biases)
        self.real_gen = np.zeros((len(self.filenames)))


        if self.split in ["train", "valid"]:

            group_counts = np.zeros((2,2))
            for target_idx in self.targets_values:
                for bias_idx in self.biases_values:
                    group_counts[target_idx, bias_idx] = np.sum((self.targets == target_idx) & (self.biases == bias_idx))
            
            min_group_positive = np.argmin(group_counts[1])
            min_group_negative = 1 - min_group_positive

            min_groups = [min_group_negative, min_group_positive]
            total_to_remove = [] 
            for target_idx in self.targets_values:
                target_min_group = min_groups[target_idx]
                target_max_group = 1 - target_min_group

                ratio_minority = 1 - 0.9
                min_bias_new_count = int((ratio_minority * group_counts[target_idx, target_max_group])/(1-ratio_minority))
                to_remove_samples_num = int(group_counts[target_idx, target_min_group] - min_bias_new_count)

                idx_to_remove = np.where((self.targets == target_idx) & (self.biases == target_min_group))[0]
                idx_to_remove = np.random.choice(idx_to_remove, to_remove_samples_num, replace=False)
                total_to_remove.extend(idx_to_remove)
            
            self.filenames = [self.filenames[i] for i in range(len(self.filenames)) if i not in total_to_remove]
            self.targets = np.delete(self.targets, total_to_remove)
            self.biases = np.delete(self.biases, total_to_remove)
            self.real_gen = np.delete(self.real_gen, total_to_remove)


    def fix_filenames(self):
        self.filenames = [os.path.join(self.root_dir, filename) for filename in self.filenames]
