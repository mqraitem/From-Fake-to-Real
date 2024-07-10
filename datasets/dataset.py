import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from imagecorruptions import corrupt

class MainDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, opt=None):

        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.opt = opt

    def get_gen_ratio(self, verbose=True): 

        num_gen = np.sum(self.real_gen == 1)
        num_real = np.sum(self.real_gen == 0)

        if verbose: 
            print("Ratio REAL/GEN: ", num_gen/num_real)
        return num_gen, num_real

    def make_harder(self, keep_min=True): 
        to_remove_samples = [] 
        for class_idx in self.targets_values: 
            class_groups = [group for group in self.GROUPS if group[0] == class_idx]
            class_groups_counts = {group:self.group_counts[group] for group in class_groups}

            #get the max key in class_groups_counts 
            max_group = max(class_groups_counts, key=class_groups_counts.get)
            min_group = min(class_groups_counts, key=class_groups_counts.get)

            ratio = 1 - self.opt.minority_to_keep
            to_keep = (class_groups_counts[max_group]*ratio)/(1 - ratio)

            if to_keep > class_groups_counts[min_group]: 
                continue 
            
            if keep_min:
                to_keep = max(to_keep, 10)
        
            to_remove_samples_num = int(class_groups_counts[min_group] - to_keep)
            idx_to_remove = np.where((self.targets == min_group[0]) & (self.biases == min_group[1]))[0]
            idx_to_remove = np.random.choice(idx_to_remove, to_remove_samples_num, replace=False)

            to_remove_samples.extend(idx_to_remove)

        self.targets = np.delete(self.targets, to_remove_samples)
        self.biases = np.delete(self.biases, to_remove_samples)
        self.filenames = [self.filenames[i] for i in range(len(self.filenames)) if i not in to_remove_samples]
        self.real_gen = np.delete(self.real_gen, to_remove_samples)

    def get_classes(self): 
        return self.targets_values

    def get_biases(self): 
        return self.biases_values

    def group_counts_dro(self): 
        group_counts = [] 
        for idx in np.unique(self.group_idx): 
            group_counts.append(np.sum(self.group_idx == idx))
        return group_counts

    def group_counts_dro_real_gen(self):
        group_counts = [] 
        for idx in np.unique(self.group_idx_real_gen): 
            group_counts.append(np.sum(self.group_idx_real_gen == idx))
        return group_counts

    def n_groups_dro(self): 
        return len(np.unique(self.group_idx))

    def n_groups_dro_real_gen(self):
        return len(np.unique(self.group_idx_real_gen))

    def get_class_distribution(self):
        for class_value_, bias_value in self.GROUPS: 

            class_index = self.targets == class_value_
            bias_idx = self.biases == bias_value

            num_images =np.sum(class_index & bias_idx)
            #print(f'Class Value {class_value_}, Bias {bias_value}:  {num_images / np.sum(class_index):.4f}')
            print(f'Class Value {class_value_}, Bias {bias_value}:  {num_images}')

        print("====================================")

    def set_group_counts(self): 
        self.group_counts_ = np.zeros((len(self.GROUPS)))
        self.group_counts = {}
        for idx, (class_value, bias_value) in enumerate(self.GROUPS): 
        
            class_index = self.targets == class_value
            bias_idx = self.biases == bias_value

            self.group_counts_[idx] = np.sum(class_index & bias_idx)
            self.group_counts[(class_value, bias_value)] = np.sum(class_index & bias_idx)

    def create_real_gen_weights(self, quit_ = False): 

        self.real_weights_groups = np.zeros_like(self.targets, dtype=np.float32)

        probs = self.group_counts_/np.sum(self.group_counts_)
        probs[probs == 0] = 1e-10
        probs = 1/probs

        for idx, (class_value, bias_value) in enumerate(self.GROUPS): 
                
            class_index = self.targets == class_value
            bias_idx = self.biases == bias_value

            self.real_weights_groups[class_index & bias_idx] = probs[idx]

        self.group_idx = np.zeros_like(self.targets, dtype=np.int32) 
        
        class_bias_pairs = [] 
        for class_label, bias_label in zip(self.targets, self.biases):
            if (class_label, bias_label) not in class_bias_pairs:
                class_bias_pairs.append((class_label, bias_label))
        
        for idx, class_bias_pair in enumerate(class_bias_pairs):
            class_value, bias_value = class_bias_pair

            class_index = self.targets == class_value
            bias_idx = self.biases == bias_value

            self.group_idx[class_index & bias_idx] = idx

        self.group_idx_real_gen = np.zeros_like(self.targets, dtype=np.int32) 
        class_bias_real_gen_pairs = []
        for class_label, bias_label, real_gen_label in zip(self.targets, self.biases, self.real_gen):
            if (class_label, bias_label, real_gen_label) not in class_bias_real_gen_pairs:
                class_bias_real_gen_pairs.append((class_label, bias_label, real_gen_label))

        for idx, class_bias_real_gen_pair in enumerate(class_bias_real_gen_pairs):
            class_value, bias_value, real_gen_value = class_bias_real_gen_pair

            class_index = self.targets == class_value
            bias_idx = self.biases == bias_value
            real_gen_idx = self.real_gen == real_gen_value

            self.group_idx_real_gen[class_index & bias_idx & real_gen_idx] = idx

        self.bias_real_gen = np.zeros_like(self.targets, dtype=np.int32)
        bias_real_gen_pairs = []

        for bias_label, real_gen_label in zip(self.biases, self.real_gen):
            if (bias_label, real_gen_label) not in bias_real_gen_pairs:
                bias_real_gen_pairs.append((bias_label, real_gen_label))
        
        for idx, bias_real_gen_pair in enumerate(bias_real_gen_pairs):
            bias_value, real_gen_value = bias_real_gen_pair

            bias_idx = self.biases == bias_value
            real_gen_idx = self.real_gen == real_gen_value

            self.bias_real_gen[bias_idx & real_gen_idx] = idx

    def load_gen_data(self):

        filenames_all = []
        targets_all = []
        biases_all = [] 

        for idx, (target_value, bias_value) in enumerate(self.GROUPS): 
            to_load_path = os.path.join(self.gen_dir, self.get_gen_dir(target_value, bias_value))
            
            filenames = os.listdir(to_load_path)
            filenames = [os.path.join(to_load_path, filename) for filename in filenames]

            start_idx = 0
            end_idx = self.opt.num_per_group 
            
            filenames_all.extend(filenames[start_idx:end_idx])
            targets_all.extend([target_value] * len(filenames[start_idx:end_idx]))
            biases_all.extend([bias_value] * len(filenames[start_idx:end_idx]))

        self.filenames.extend(filenames_all)
        self.targets = np.concatenate((self.targets, np.array(targets_all)), axis=0)
        self.biases = np.concatenate((self.biases, np.array(biases_all)), axis=0)
        self.real_gen = np.concatenate((self.real_gen, np.ones(len(targets_all))), axis=0)

        self.set_group_counts()
        self.create_real_gen_weights()

    def load_gen_data_balance(self):

        to_add_all_filenames = []
        to_add_all_targets = [] 
        to_add_all_biases = [] 
        to_add_all_real_gen = [] 

        max_count = np.max(self.group_counts_) 
        for idx, (target_value, bias_value) in enumerate(self.GROUPS): 

            to_add = int(max_count - self.group_counts_[idx])
            to_load_path = os.path.join(self.gen_dir, self.get_gen_dir(target_value, bias_value))

            filenames = os.listdir(to_load_path)
            filenames = [os.path.join(to_load_path, filename) for filename in filenames]
            filenames = filenames[:to_add]

            to_add_all_targets.extend([target_value] * len(filenames))
            to_add_all_biases.extend([bias_value] * len(filenames))
            to_add_all_real_gen.extend([1] * len(filenames))
            to_add_all_filenames.extend(filenames)

        self.filenames.extend(to_add_all_filenames)
        self.targets = np.concatenate((self.targets, np.array(to_add_all_targets)), axis=0)
        self.biases = np.concatenate((self.biases, np.array(to_add_all_biases)), axis=0)
        self.real_gen = np.concatenate((self.real_gen, np.array(to_add_all_real_gen)), axis=0)

        self.set_group_counts()
        self.create_real_gen_weights()
        
    def load_image(self, filename):
        img = Image.open(filename)
        img = img.convert('RGB')
        return img

    def get_target_bias_to_ratio(self):
        target_bias_to_ratio = {} 
        for _, (target_value, bias_value) in enumerate(self.GROUPS): 
            idx = np.where((self.targets == target_value) & (self.biases == bias_value))[0]
            idx_targets = np.where(self.targets == target_value)[0]
            target_bias_to_ratio[(target_value, bias_value)] = len(idx)/len(idx_targets)
        
        return target_bias_to_ratio

    def corrupt_image(self, img):
        img = np.asarray(img)
        img = corrupt(img, corruption_name="pixelate", severity=int(self.opt.severity))
        img = corrupt(img, corruption_name="defocus_blur", severity=int(self.opt.severity))
        img = corrupt(img, corruption_name="gaussian_noise", severity=int(self.opt.severity))

        #convert back to PIL image
        img = Image.fromarray(img)
        return img
    

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        
        filename = self.filenames[idx]
        img = self.load_image(filename)
        target = self.targets[idx]
        bias = self.biases[idx]
        
        idx_img = filename.split('/')[-1].split('.')[0]
        real_gen = self.real_gen[idx]
        
        group_idx = self.group_idx[idx]
        real_gen_weights = self.real_weights_groups[idx]
        group_idx_real_gen = self.group_idx_real_gen[idx]
        bias_real_gen = self.bias_real_gen[idx]

        if ("severity" in self.opt) and (self.split == "train"): 
            if self.opt.severity > 0 and real_gen == 1:
                img = self.corrupt_image(img)

        if self.transform:
            img = self.transform(img)


        data = {
            'idx': idx,
            'img': img,
            'target': target,
            'bias': bias,
            'filenames': filename, 
            'real_gen': real_gen,
            'group_idx':group_idx,
            'real_weights_groups': real_gen_weights,
            'group_idx_real_gen': group_idx_real_gen,
            'bias_real_gen': bias_real_gen

        }

        return data
