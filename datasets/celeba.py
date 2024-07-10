import os
import numpy as np
import pandas as pd
import pickle
import sys
from collections import namedtuple
from datasets.dataset import MainDataset

CSV = namedtuple("CSV", ["header", "index", "data"])

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


class CelebA(MainDataset):
    def __init__(self, root_dir, split='train', transform=None, opt=None):
        super(CelebA, self).__init__(root_dir, split, transform, opt)

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

        self.targets_values = range(2)
        self.biases_values = range(2)

        self.load_filenames()
        self.load_labels()

        self.gen_dir = f"./generated_images/celeba/{opt.target_attr}"        
        
        path_file = f"data/celeba_{opt.target_attr}_{split}_{self.opt.minority_to_keep}.pkl"

        if os.path.exists(path_file):
            data = pickle.load(open(path_file, "rb"))
            self.filenames = data['filenames']
            self.targets = data['targets']
            self.biases = data['biases']
            self.real_gen = data['real_gen']

        else: 
            self.load_labels()
            self.fix_filenames()
            self.set_group_counts()

            if split == 'train':
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
            pickle.dump(data, open(path_file, "wb"))

        self.set_group_counts()
        self.create_real_gen_weights()


    def get_gen_dir(self, target_value, bias_value):
        return f"{target_value}/{self.bias_names[bias_value]}" 

    def load_filenames(self):
        self.filenames = os.listdir(os.path.join(self.root_dir, "CelebA-HQ-img"))
        start_idx = int(len(self.filenames) * self.split_ratio[self.split][0])
        end_idx = int(len(self.filenames) * self.split_ratio[self.split][1])

        self.filenames = self.filenames[start_idx:end_idx]

    def fix_filenames(self): 
        self.filenames = [os.path.join(self.root_dir, 'CelebA-HQ-img', filename) for filename in self.filenames]

    def load_labels(self):
        self.labels = pd.read_csv(os.path.join(self.root_dir, 'CelebAMask-HQ-attribute-anno.txt'), delim_whitespace=True, skiprows=1, index_col=0)
        self.class_names = list(self.labels.columns.values)

        self.labels = self.labels.loc[self.filenames]
        self.labels = self.labels.to_numpy()
        self.labels[self.labels == -1] = 0
        
        self.targets = self.labels[:, self.class_names.index(self.opt.target_attr)]
        self.biases = self.labels[:, 20] #index Male
        self.bias_names = ["Female", "Male"]

        self.real_gen = np.zeros((len(self.filenames)))

        self.targets = self.targets.astype(int)
        self.real_gen = self.real_gen.astype(int)

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
