import os
import torch
import numpy as np
import pandas as pd

import sys
sys.path.insert(1, './')

from torch.optim import Adam

from tqdm import tqdm
from utils import set_seed, compute_stats
from torchvision import transforms as T
from utils import ResNet50

class TrainerBase():    
    def __init__(self, root_dir, transform=None, opt=None):
        self.root_dir = root_dir
        self.transform = transform
        self.batch_size = opt.batch_size
        self.opt = opt

        self.epochs = opt.epochs
        self.lr = opt.lr
        set_seed(opt.seed)

    def load_model(self):
        self.model = ResNet50(num_classes=len(self.dataset_train.targets_values))
        self.model.to('cuda')

        if self.opt.pretrain_dir != 'None': 
            print(f'Loading pretrained model from {self.opt.pretrain_dir}')
            self.model.load_state_dict(torch.load(f"{self.opt.pretrain_dir}/best_model_{self.opt.seed}.pth"))
            self.model.to('cuda')

        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.opt.weight_decay)
        self.best_accuracy = -np.inf


    def get_best_hyper(self, dir): 
        best_acc = 0
        best_hyper = None

        for hyper in os.listdir(dir): 
            hyper_acc = 0 
            for seed_file in os.listdir(os.path.join(dir, hyper)): 
                df = pd.read_csv(os.path.join(dir, hyper, seed_file))
                hyper_acc +=  df["worst_acc_val"].iloc[0]
            hyper_acc = hyper_acc/len(os.listdir(os.path.join(dir, hyper)))
            if hyper_acc > best_acc:
                best_acc = hyper_acc
                best_hyper = hyper
        
        return best_hyper

    def evaluate(self, dataloader, model): 
        self.model.eval()
        biases = dataloader.dataset.get_biases()

        output_list = [] 
        target_list = [] 
        bias_list = []

        with torch.no_grad():
            for _, data in enumerate(tqdm(dataloader, ascii=True)):
                img = data['img'].cuda()
                target = data['target']
                bias = data['bias']

                output = model(img).detach().cpu()
                output_list.append(output) 
                target_list.append(target)
                bias_list.append(bias)

        outputs = torch.cat(output_list, dim=0).numpy()
        targets = torch.cat(target_list, dim=0).numpy()
        biases = torch.cat(bias_list, dim=0).numpy() 

        worst_acc, balanced_acc = compute_stats(outputs, targets, biases, self.dataset_train.GROUPS)
        
        print(f'Worst acc: {worst_acc}')
        print(f'Balanced acc: {balanced_acc}')

        return worst_acc, balanced_acc

    def test(self, split='test'):
        self.model.load_state_dict(torch.load(self.model_save_name))

        worst_acc_val, balanced_acc_val = self.evaluate(self.dataloader_val, self.model)
        worst_acc_test, balanced_acc_test = self.evaluate(self.dataloader_test, self.model)

        results = { 
            "worst_acc_val": worst_acc_val,
            "balanced_acc_val": balanced_acc_val,
            "worst_acc_test": worst_acc_test,
            "balanced_acc_test": balanced_acc_test,
        }

        df = pd.DataFrame(results, index=[0])
        df.to_csv(os.path.join(self.results_dir, f'results_{self.opt.seed}.csv'))

