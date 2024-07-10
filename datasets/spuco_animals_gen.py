import numpy as np
from datasets.dataset_gen import MainDatasetGen
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


class SpucoAnimalsGen(MainDatasetGen):
    def __init__(self, root_dir, split='train', transform=None, opt=None):

        super(SpucoAnimalsGen, self).__init__(root_dir, split, transform, opt)
        self.GROUPS =[(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (2, 3), (3, 2), (3, 3)]

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

        self.gen_dir = f"./generated_images/spuco_animals"        

        self.load_gen_data()
        self.set_group_counts()

        self.targets_values = range(4)
        self.biases_values = range(4)
        
    def get_gen_dir(self, target_value, bias_value):
        return f"{self.class_to_name[target_value]}_{self.bias_to_name[bias_value]}"

