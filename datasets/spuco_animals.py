import os
import numpy as np
import pickle
import sys
from collections import namedtuple
from datasets.dataset import MainDataset

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)

SMALL_DOGS = "small_dogs"
BIG_DOGS = "big_dogs"
INDOOR = "indoor"
OUTDOOR = "outdoor"

LANDBIRDS = "landbirds"
WATERBIRDS = "waterbirds"
LAND = "land"
WATER = "water"

MAJORITY_SIZE = {
    "train": 10000,
    "val": 500,
    "test": 500,
}
MINORITY_SIZE = {
    "train": 500,
    "val": 25,
    "test": 500,
}


class SpucoAnimals(MainDataset):
    def __init__(self, root_dir, split='train', transform=None, opt=None):
        super(SpucoAnimals, self).__init__(root_dir, split, transform, opt)

        self.GROUPS = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (2, 3), (3, 2), (3, 3)]

        self.target_attr = "gender"
        self.bias_attr = "age"

        self.class_to_name = {
            0:"landbird", 
            1:"waterbird",
            2:"small_dog", 
            3:"big_dog"
        }

        self.bias_to_name = { 
            0:"Land", 
            1:"Water", 
            2:"Indoors", 
            3:"Outdoors",
        }

        self.targets_values = range(4)
        self.biases_values = range(4)

        self.gen_dir = f"./generated_images/spuco_animals"        
    
        if os.path.exists(f"data/spuco_animals_{split}_{self.opt.minority_to_keep}.pkl"):
            data = pickle.load(open(f"data/spuco_animals_{split}_{self.opt.minority_to_keep}.pkl", "rb"))
            self.filenames = data['filenames']
            self.targets = data['targets']
            self.biases = data['biases']
            self.real_gen = data['real_gen']

        else: 
        
            self.load_data()
            self.set_group_counts()

            if split == "train":
                self.make_harder(False)
                self.set_group_counts()

            if split == 'val':
                self.make_harder()
                self.set_group_counts()

            data = {
                'filenames': self.filenames,
                'targets': self.targets,
                'biases': self.biases,
                'real_gen': self.real_gen
            }
            pickle.dump(data, open(f"data/spuco_animals_{split}_{self.opt.minority_to_keep}.pkl", "wb"))

        self.set_group_counts()
        self.create_real_gen_weights()
        # self.get_min_groups()


    def get_gen_dir(self, target_value, bias_value):
        return f"{self.class_to_name[target_value]}_{self.bias_to_name[bias_value]}" 

    def load_dogs(self): 

        filenames = [] 
        targets = []
        biases = []

        root_dir_birds =  os.path.join(self.root_dir, "spuco_dogs", self.split)

        small_dogs_indoor = os.listdir(os.path.join(root_dir_birds, f"{SMALL_DOGS}/{INDOOR}"))
        filenames.extend([str(os.path.join(root_dir_birds, f"{SMALL_DOGS}/{INDOOR}", x)) for x in small_dogs_indoor])
        targets.extend([0] * len(small_dogs_indoor))
        biases.extend([0] * len(small_dogs_indoor))
        assert len(small_dogs_indoor) == MAJORITY_SIZE[self.split], f"Dataset corrupted or missing files. Expected {MAJORITY_SIZE[self.split]} files got {len(small_dogs_indoor)}"
        
        # Small Dogs - Outdoor
        small_dogs_outdoor = os.listdir(os.path.join(root_dir_birds, f"{SMALL_DOGS}/{OUTDOOR}"))
        filenames.extend([str(os.path.join(root_dir_birds, f"{SMALL_DOGS}/{OUTDOOR}", x)) for x in small_dogs_outdoor])
        targets.extend([0] * len(small_dogs_outdoor))
        biases.extend([1] * len(small_dogs_outdoor))   
        assert len(small_dogs_outdoor) == MINORITY_SIZE[self.split], f"Dataset corrupted or missing files. Expected {MINORITY_SIZE[self.split]} files got {len(small_dogs_outdoor)}"
        
        # Big Dogs - Indoor
        big_dogs_indoor = os.listdir(os.path.join(root_dir_birds, f"{BIG_DOGS}/{INDOOR}"))
        filenames.extend([str(os.path.join(root_dir_birds, f"{BIG_DOGS}/{INDOOR}", x)) for x in big_dogs_indoor])
        targets.extend([1] * len(big_dogs_indoor))
        biases.extend([0] * len(big_dogs_indoor))
        assert len(big_dogs_indoor) == MINORITY_SIZE[self.split], f"Dataset corrupted or missing files. Expected {MINORITY_SIZE[self.split]} files got {len(big_dogs_indoor)}"
        
        # Big Dogs - Outdoor
        big_dogs_outdoor = os.listdir(os.path.join(root_dir_birds, f"{BIG_DOGS}/{OUTDOOR}"))
        filenames.extend([str(os.path.join(root_dir_birds, f"{BIG_DOGS}/{OUTDOOR}", x)) for x in big_dogs_outdoor])
        targets.extend([1] * len(big_dogs_outdoor))
        biases.extend([1] * len(big_dogs_outdoor)) 
        assert len(big_dogs_outdoor) == MAJORITY_SIZE[self.split], f"Dataset corrupted or missing files. Expected {MAJORITY_SIZE[self.split]} files got {len(big_dogs_outdoor)}"

        return filenames, targets, biases

    def load_birds(self): 

        filenames = [] 
        targets = []
        biases = []

        root_dir_birds =  os.path.join(self.root_dir, "spuco_birds", self.split)


        landbirds_land = os.listdir(os.path.join(root_dir_birds, f"{LANDBIRDS}/{LAND}"))
        filenames.extend([str(os.path.join(root_dir_birds, f"{LANDBIRDS}/{LAND}", x)) for x in landbirds_land])
        targets.extend([0] * len(landbirds_land))
        biases.extend([0] * len(landbirds_land))
        assert len(landbirds_land) == MAJORITY_SIZE[self.split], f"Dataset corrupted or missing files [landbirds_land]. Expected {MAJORITY_SIZE[self.split]} files got {len(landbirds_land)}"
        
        # Landbirds Water 
        landbirds_water = os.listdir(os.path.join(root_dir_birds, f"{LANDBIRDS}/{WATER}"))
        filenames.extend([str(os.path.join(root_dir_birds, f"{LANDBIRDS}/{WATER}", x)) for x in landbirds_water])
        targets.extend([0] * len(landbirds_water))
        biases.extend([1] * len(landbirds_water))   
        assert len(landbirds_water) == MINORITY_SIZE[self.split], f"Dataset corrupted or missing files [landbirds_water]. Expected {MINORITY_SIZE[self.split]} files got {len(landbirds_water)}"
        
        # Waterbirds Land
        waterbirds_land = os.listdir(os.path.join(root_dir_birds, f"{WATERBIRDS}/{LAND}"))
        filenames.extend([str(os.path.join(root_dir_birds, f"{WATERBIRDS}/{LAND}", x)) for x in waterbirds_land])
        targets.extend([1] * len(waterbirds_land))
        biases.extend([0] * len(waterbirds_land))
        assert len(waterbirds_land) == MINORITY_SIZE[self.split], f"Dataset corrupted or missing files [waterbirds_land]. Expected {MINORITY_SIZE[self.split]} files got {len(waterbirds_land)}"
        
        # Waterbirds Water
        waterbirds_water = os.listdir(os.path.join(root_dir_birds, f"{WATERBIRDS}/{WATER}"))
        filenames.extend([str(os.path.join(root_dir_birds, f"{WATERBIRDS}/{WATER}", x)) for x in waterbirds_water])
        targets.extend([1] * len(waterbirds_water))
        biases.extend([1] * len(waterbirds_water)) 
        assert len(waterbirds_water) == MAJORITY_SIZE[self.split], f"Dataset corrupted or missing files [waterbirds_water]. Expected {MAJORITY_SIZE[self.split]} files got {len(waterbirds_water)}"

        return filenames, targets, biases

    def load_data(self): 

        self.filenames = []
        self.targets = []
        self.biases = []
        
        dogs_filenames, dogs_targets, dogs_biases = self.load_dogs()
        birds_filenames, birds_targets, birds_biases = self.load_birds()

        self.filenames.extend(birds_filenames)
        self.targets.extend(birds_targets)
        self.biases.extend(birds_biases)

        self.filenames.extend(dogs_filenames)
        self.targets.extend([label + 2 for label in dogs_targets])
        self.biases.extend([bias + 2 for bias in dogs_biases])

        self.filenames = np.array(self.filenames)
        self.targets = np.array(self.targets)
        self.biases = np.array(self.biases)
        self.real_gen = np.zeros_like(self.targets)


# def get_remove_idx(self, min_bias, counts_class, class_idx): 
#         max_bias = 1 - min_bias 
    
#         ratio_minority = 1 - self.opt.minority_to_keep 
#         min_bias_new_count = int((ratio_minority * counts_class[max_bias])/(1-ratio_minority))
#         to_remove_samples_num = int(counts_class[min_bias] - min_bias_new_count)

#         idx_to_remove = np.where((self.targets == class_idx) & (self.biases == min_bias))[0]
#         idx_to_remove = np.random.choice(idx_to_remove, to_remove_samples_num, replace=False)

#         return idx_to_remove



# def get_min_groups(self): 

#     counts = []
#     target_to_bias = {x:[] for x in self.targets_values}
#     for target_value, bias_value in GROUPS:
#         target_to_bias[target_value].append(bias_value)
    
#     target_to_min_bias = {x:[] for x in self.targets_values}
#     for target_value, bias_values in target_to_bias.items():

#         idx_min = np.argmin([np.sum((self.targets == target_value) & (self.biases == bias_value)) for bias_value in bias_values])
#         target_to_min_bias[target_value] = target_to_bias[target_value][idx_min]

#     self.target_to_min_bias = target_to_min_bias    



# def make_harder(self, min_group_count, max_group_count, keep_min = True): 

#     to_remove_samples_total = [] 

#     for class_idx in self.targets_values: 
#         for idx, (class_idx_group, bias_idx_group) in enumerate(GROUPS): 
            
#             if class_idx_group != class_idx: 
#                 continue
            
#             count_group = self.group_counts_[idx] 
#             if count_group > min_group_count: 
#                 continue
            
#             ratio = 1 - self.opt.minority_to_keep
#             to_keep = (max_group_count*ratio)/(1 - ratio)

#             if keep_min: 
#                 to_keep = max(to_keep, 15)
#             to_remove_samples_num = int(count_group - to_keep)
            
#             idx_to_remove =  np.where((self.targets == class_idx_group) & (self.biases == bias_idx_group))[0]
#             idx_to_remove = np.random.choice(idx_to_remove, to_remove_samples_num, replace=False)

#             to_remove_samples_total.extend(idx_to_remove)

#     self.targets = np.delete(self.targets, to_remove_samples_total)
#     self.biases = np.delete(self.biases, to_remove_samples_total)
#     self.filenames = list(np.delete(np.array(self.filenames), to_remove_samples_total))

#     self.real_gen = np.delete(self.real_gen, to_remove_samples_total)

