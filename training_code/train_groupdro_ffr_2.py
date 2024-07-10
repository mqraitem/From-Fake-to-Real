import os
import torch
import numpy as np
from torchvision import transforms
import pandas as pd

import sys
sys.path.insert(1, './')

from torch.optim import Adam

from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from utils import set_seed, compute_stats
from torchvision import transforms as T
from utils import DROLossComputer
from utils import ResNet50

from trainer import TrainerBase
from datasets.celeba import CelebA
from datasets.utk_face import UtkFace
from datasets.spuco_animals import SpucoAnimals

from utils import get_datasets_dir

NAME_TO_DATASET = { 
    "celeba": CelebA,
    "utk_face": UtkFace,
    "spuco_animals": SpucoAnimals,
}

class ResNet50Trainer(TrainerBase):    
    def __init__(self, root_dir, transform=None, opt=None):
        super(ResNet50Trainer, self).__init__(root_dir, transform, opt)

        self.epochs = opt.epochs
        self.lr = opt.lr
        set_seed(opt.seed)

        #load the datasets for train/test/val 
        valid_name = "val" if opt.dataset == "spuco_animals" else "valid"
        self.dataset_train = NAME_TO_DATASET[opt.dataset](root_dir, split='train', transform=transform["train"], opt=opt)
        self.dataset_val = NAME_TO_DATASET[opt.dataset](root_dir, split=valid_name, transform=transform["eval"], opt=opt)
        self.dataset_test = NAME_TO_DATASET[opt.dataset](root_dir, split='test', transform=transform["eval"], opt=opt)

        self.dataloader_train = DataLoader(self.dataset_train, batch_size=opt.batch_size, shuffle=True, num_workers=2)
        self.dataloader_val = DataLoader(self.dataset_val, batch_size=opt.batch_size, num_workers=2)
        self.dataloader_test = DataLoader(self.dataset_test, batch_size=opt.batch_size, num_workers=2)

        self.dataset_train.get_class_distribution()
        self.dataset_val.get_class_distribution()
        self.dataset_test.get_class_distribution()

        gen_dir = "ratio_balance"
        opt.method = 'groupdro_ffr_2'

        pretrain_dir = f"{opt.dataset}/{opt.minority_to_keep}/erm_ffr_1/{gen_dir}/"
        pretrain_dir_model = f"output/models/{pretrain_dir}"
        pretrain_dir_results = f"output/results/{pretrain_dir}"

        best_hyper = self.get_best_hyper(pretrain_dir_results)
        opt.pretrain_dir = f"{pretrain_dir_model}/{best_hyper}"
            
        self.model_dir = f'output/models/{opt.dataset}/{opt.minority_to_keep}/{opt.method}/{gen_dir}/{opt.lr}_{opt.weight_decay}_{opt.freeze_bn}/'
        self.results_dir = f'output/results/{opt.dataset}/{opt.minority_to_keep}/{opt.method}/{gen_dir}/{opt.lr}_{opt.weight_decay}_{opt.freeze_bn}/'
        self.model_save_name = f'{self.model_dir}/best_model_{opt.seed}.pth'

        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        if os.path.exists(os.path.join(self.results_dir, f'results_{self.opt.seed}.csv')): 
            print("Done")
            quit()

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

    def load_model(self):
        self.model = ResNet50(num_classes=len(self.dataset_train.targets_values))
        if self.opt.pretrain_dir != 'None': 
            print(f'Loading pretrained model from {self.opt.pretrain_dir}')
            self.model.load_state_dict(torch.load(f"{self.opt.pretrain_dir}/best_model_{self.opt.seed}.pth"))

        self.model.to('cuda')
        adjustments = [float(c) for c in self.opt.generalization_adjustment.split(',')]
        if len(adjustments)==1:
            adjustments = np.array(adjustments* self.dataset_train.n_groups_dro())
        else:
            adjustments = np.array(adjustments)

        self.criterion = DROLossComputer(
            torch.nn.CrossEntropyLoss(reduction='none'),
            is_robust=self.opt.robust,
            n_groups=self.dataset_train.n_groups_dro(),
            group_counts=self.dataset_train.group_counts_dro(),
            alpha=self.opt.alpha,
            gamma=self.opt.gamma,
            adj=adjustments,
            step_size=self.opt.robust_step_size,
            normalize_loss=self.opt.use_normalized_loss,
            btl=self.opt.btl,
            min_var_weight=self.opt.minimum_variational_weight)
        
        self.optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.opt.weight_decay)
        self.best_accuracy = -np.inf

    def train(self):
        if self.opt.freeze_bn:
            self.model.eval()        
        else: 
            self.model.train()

        for epoch in range(self.epochs):
            for i, data in enumerate(tqdm(self.dataloader_train, ascii=True)):
                self.optimizer.zero_grad()
                img, label = data['img'].to('cuda'), data['target'].to('cuda')
                group_idx = data['group_idx'].to('cuda')

                output = self.model(img)  

                loss = self.criterion.loss(output, label, group_idx, True)
                loss.backward()
                self.optimizer.step()
                if i % 100 == 0:
                    print('Epoch: {} Iteration: {} Loss: {}'.format(epoch, i, loss))

            accuracy, _ = self.evaluate(self.dataloader_val, self.model)

            if self.opt.freeze_bn:
                self.model.eval()
            else:
                self.model.train()

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy

                
                torch.save(self.model.state_dict(), self.model_save_name)            



if __name__ == '__main__':
    #define the transform
    transform = {
        "train": transforms.Compose([
        T.Resize(size=(224, 224)),
        # T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        "eval":  transforms.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='celeba')
    parser.add_argument('--target_attr', type=str, default='Smiling')

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--minority_to_keep', type=float, default=0.95)

    parser.add_argument('--robust', default=False, action='store_true')
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--generalization_adjustment', default="0.0")
    parser.add_argument('--automatic_adjustment', default=False, action='store_true')
    parser.add_argument('--robust_step_size', default=0.01, type=float)
    parser.add_argument('--use_normalized_loss', default=False, action='store_true')
    parser.add_argument('--btl', default=False, action='store_true')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--minimum_variational_weight', type=float, default=0)
    parser.add_argument('--freeze_bn', action='store_true')
    parser.add_argument('--limit_to_gen', action='store_true')

    opt = parser.parse_args()
    root_dir = get_datasets_dir(opt.dataset)

    trainer = ResNet50Trainer(root_dir=root_dir, transform=transform, opt=opt)
    #load the model
    trainer.load_model()
    #train the model
    trainer.train()
    trainer.test()
