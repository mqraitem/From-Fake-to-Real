import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import sys
from imagecorruptions import corrupt

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)

class MainDatasetGen(Dataset):
    def __init__(self, root_dir, split='train', transform=None, opt=None):

        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.opt = opt

    def get_classes(self): 
        return self.targets_values

    def get_biases(self): 
        return self.biases_values

    def group_counts(self): 
        return self.group_counts_

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

    def load_gen_data(self):

        filenames_all = []
        targets_all = []
        biases_all = [] 

        for idx, (target_value, bias_value) in enumerate(self.GROUPS): 
            to_load_path = os.path.join(self.gen_dir, self.get_gen_dir(target_value, bias_value))
            
            filenames = os.listdir(to_load_path)
            filenames = [os.path.join(to_load_path, filename) for filename in filenames]

            start_idx = int(len(filenames) * self.split_ratio[self.split][0])
            end_idx = int(len(filenames) * self.split_ratio[self.split][1])

            if self.split == "train": 
                end_idx = self.opt.num_per_group 
            
            filenames_all.extend(filenames[start_idx:end_idx])
            targets_all.extend([target_value] * len(filenames[start_idx:end_idx]))
            biases_all.extend([bias_value] * len(filenames[start_idx:end_idx]))

        self.filenames = filenames_all
        self.targets = np.array(targets_all)
        self.biases = np.array(biases_all)

    def load_gen_data_biased(self, target_bias_to_load): 

        filenames_all = []
        targets_all = []
        biases_all = [] 

        for idx, (target_value, bias_value) in enumerate(self.GROUPS): 
            to_load_path = os.path.join(self.gen_dir, self.get_gen_dir(target_value, bias_value))
            
            filenames = os.listdir(to_load_path)
            filenames = [os.path.join(to_load_path, filename) for filename in filenames]

            start_idx = int(len(filenames) * self.split_ratio[self.split][0])
            end_idx = int(len(filenames) * self.split_ratio[self.split][1])

            if self.split == "train": 
                bias_ratio = target_bias_to_load[(target_value, bias_value)]
                end_idx = int((self.opt.num_per_group * 2) * bias_ratio)

            filenames_all.extend(filenames[start_idx:end_idx])
            targets_all.extend([target_value] * len(filenames[start_idx:end_idx]))
            biases_all.extend([bias_value] * len(filenames[start_idx:end_idx]))

        self.filenames = filenames_all
        self.targets = np.array(targets_all)
        self.biases = np.array(biases_all)


    def load_image(self, filename):
        img = Image.open(filename)
        return img

    def get_class_distribution(self):
        for class_value_ in range(2): 
            for bias_value in self.biases_values: 

                class_index = self.targets == class_value_
                bias_idx = self.biases == bias_value

                num_images =np.sum(class_index & bias_idx)
                # print(f'Class Value {class_value_}, Bias {bias_value}:  {num_images / np.sum(class_index):.4f}')
                print(f'Class Value {class_value_}, Bias {bias_value}:  {num_images:.4f}')


        print("====================================")

    def set_group_counts(self): 
        self.group_counts_ = np.zeros((2,2))
        for class_value in range(2): 
            for bias_value in range(2): 
                class_index = self.targets == class_value
                bias_idx = self.biases == bias_value

                self.group_counts_[class_value, bias_value] = np.sum(class_index & bias_idx)


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
        target = self.targets[idx]
        bias = self.biases[idx]
        idx_img = filename.split('/')[-1].split('.')[0]
        img = self.load_image(filename)

        if self.opt.severity > 0: 
            img = self.corrupt_image(img)

        if self.transform:
            img = self.transform(img)

        data = {
            'idx': idx,
            'img': img,
            'target': target,
            'bias': bias,
            'filenames': filename, 
        }

        return data




# def load_gen_data_blind(self):


#     self.gen_dir = f"./generated_images_blind/{self.opt.target_attr}"        


#     filenames_all = []
#     targets_all = []
#     biases_all = [] 

#     for target_value in range(2): 
#         to_load_path = os.path.join(self.gen_dir, str(target_value))
        
#         filenames = os.listdir(to_load_path)
#         filenames = [os.path.join(to_load_path, filename) for filename in filenames]

#         start_idx = int(len(filenames) * self.split_ratio[self.split][0])
#         end_idx = int(len(filenames) * self.split_ratio[self.split][1])

#         if self.split == "train": 
#             end_idx = self.opt.num_per_group * 2
        
#         filenames_all.extend(filenames[start_idx:end_idx])
#         targets_all.extend([target_value] * len(filenames[start_idx:end_idx]))
#         biases_all.extend([0] * len(filenames[start_idx:end_idx]))

#     self.filenames = filenames_all
#     self.targets = np.array(targets_all)
#     self.biases = np.array(biases_all)


# def load_gen_data_biased(self, target_bias_to_load): 

#     filenames_all = []
#     targets_all = []
#     biases_all = [] 

#     for target_value in range(2):
#         for bias_value in range(2): 

#             to_load_path = os.path.join(self.gen_dir, str(target_value), self.bias_names[bias_value])
            
#             filenames = os.listdir(to_load_path)
#             filenames = [os.path.join(to_load_path, filename) for filename in filenames]

#             start_idx = int(len(filenames) * self.split_ratio[self.split][0])
#             end_idx = int(len(filenames) * self.split_ratio[self.split][1])

#             if self.split == "train": 
#                 bias_ratio = target_bias_to_load[(target_value, bias_value)]
#                 end_idx = int((self.opt.num_per_group * 2) * bias_ratio)

#             filenames_all.extend(filenames[start_idx:end_idx])
#             targets_all.extend([target_value] * len(filenames[start_idx:end_idx]))
#             biases_all.extend([bias_value] * len(filenames[start_idx:end_idx]))

#     self.filenames = filenames_all
#     self.targets = np.array(targets_all)
#     self.biases = np.array(biases_all)
