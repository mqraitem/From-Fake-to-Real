import os
import torch
import numpy as np
from torchvision import transforms

import sys
sys.path.insert(1, './')

from torch.optim import Adam

from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from utils import set_seed
from torchvision import transforms as T
from utils import DROLossComputer
from utils import ResNet50

from datasets.celeba import CelebA
from datasets.celeba_gen import CelebAGen
from datasets.utk_face import UtkFace
from datasets.utk_face_gen import UtkFaceGen
from datasets.spuco_animals import SpucoAnimals
from datasets.spuco_animals_gen import SpucoAnimalsGen

from trainer import TrainerBase
from utils import get_datasets_dir

NAME_TO_DATASET = { 
    "celeba": CelebA,
    "celeba_gen": CelebAGen,
    "utk_face": UtkFace,
    "utk_face_gen": UtkFaceGen,
    "spuco_animals": SpucoAnimals,
    "spuco_animals_gen": SpucoAnimalsGen,
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
        
        dataset_train_balance = NAME_TO_DATASET[opt.dataset](root_dir, split='train', transform=transform["train"], opt=opt)
        dataset_train_balance.load_gen_data_balance()
        num_gen, num_real = dataset_train_balance.get_gen_ratio(False)
        opt.num_per_group = num_gen // len(self.dataset_train.GROUPS) 
        self.dataset_train.load_gen_data()
        gen_dir = f'ratio_balance'
        opt.pretrain_dir = 'None'

        self.dataset_val = NAME_TO_DATASET[opt.dataset](root_dir, split=valid_name, transform=transform["eval"], opt=opt)
        self.dataset_test = NAME_TO_DATASET[opt.dataset](root_dir, split='test', transform=transform["eval"], opt=opt)
        
        self.dataloader_train = DataLoader(self.dataset_train, batch_size=opt.batch_size, shuffle=True, num_workers=2)
        self.dataloader_val = DataLoader(self.dataset_val, batch_size=opt.batch_size, num_workers=2)
        self.dataloader_test = DataLoader(self.dataset_test, batch_size=opt.batch_size, num_workers=2)

        self.dataset_train.get_class_distribution()
        self.dataset_val.get_class_distribution()
        self.dataset_test.get_class_distribution()

        #format num_gen/num_real up to two decimals 
        opt.method = 'groupdro_asb'

        self.model_dir = f'output/models/{opt.dataset}/{opt.minority_to_keep}/{opt.method}/{gen_dir}/{opt.lr}_{opt.weight_decay}/'
        self.model_save_name = f'{self.model_dir}/best_model_{opt.seed}.pth'
        self.results_dir = f'output/results/{opt.dataset}/{opt.minority_to_keep}/{opt.method}/{gen_dir}/{opt.lr}_{opt.weight_decay}/'

        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        if os.path.exists(os.path.join(self.results_dir, f'results_{self.opt.seed}.csv')): 
            print("Done")
            quit()

    def load_model(self):
        self.model = ResNet50(num_classes=len(self.dataset_train.targets_values))
        if self.opt.pretrain_dir != 'None': 
            print(f'Loading pretrained model from {self.opt.pretrain_dir}')
            self.model.load_state_dict(torch.load(self.opt.pretrain_dir))

        self.model.to('cuda')
        adjustments = [float(c) for c in self.opt.generalization_adjustment.split(',')]
        if len(adjustments)==1:
            adjustments = np.array(adjustments* self.dataset_train.n_groups_dro_real_gen())
        else:
            adjustments = np.array(adjustments)

        self.criterion = DROLossComputer(
            torch.nn.CrossEntropyLoss(reduction='none'),
            is_robust=self.opt.robust,
            n_groups=self.dataset_train.n_groups_dro_real_gen(),
            group_counts=self.dataset_train.group_counts_dro_real_gen(),
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
        self.model.train()
        for epoch in range(self.epochs):
            for i, data in enumerate(tqdm(self.dataloader_train, ascii=True)):
                self.optimizer.zero_grad()
                img, label = data['img'].to('cuda'), data['target'].to('cuda')
                group_idx = data['group_idx_real_gen'].to('cuda')

                output = self.model(img)  
                loss = self.criterion.loss(output, label, group_idx, True)
                loss.backward()
                self.optimizer.step()
                if i % 100 == 0:
                    print('Epoch: {} Iteration: {} Loss: {}'.format(epoch, i, loss))

            accuracy, _ = self.evaluate(self.dataloader_val, self.model)
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
    parser.add_argument('--lr', type=float, default=1e-4)
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

    opt = parser.parse_args()
    root_dir = get_datasets_dir(opt.dataset)

    #define the trainer
    trainer = ResNet50Trainer(root_dir=root_dir, transform=transform, opt=opt)
    #load the model
    trainer.load_model()
    #train the model
    trainer.train()
    trainer.test()
