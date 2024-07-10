import numpy as np
from datasets.dataset_gen import MainDatasetGen
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)

class UtkFaceGen(MainDatasetGen):
    def __init__(self, root_dir, split='train', transform=None, opt=None):

        super(UtkFaceGen, self).__init__(root_dir, split, transform, opt)

        self.GROUPS =[(0, 0), (0, 1), (1, 0), (1, 1)]
        self.split_idx = { 
            'train': 0,
            'valid': 1,
            'test': 2
        }
        self.split_ratio = {
            'train': [0.0, 0.9],
            'valid': [0.9, 0.95],
            'test':  [0.95, 1.0],
        }

        self.target_attr = "gender"
        self.bias_attr = "age"

        self.class_names = ["Male", "Female"]
        self.bias_names = ["old", "young"]

        self.gen_dir = f"./generated_images/utk_face"        

        self.load_gen_data()
        self.set_group_counts()

        self.targets_values = range(2)
        self.biases_values = range(2)
        
    def get_gen_dir(self, target_value, bias_value):
        return f"{self.class_names[target_value]}/{self.bias_names[bias_value]}" 
