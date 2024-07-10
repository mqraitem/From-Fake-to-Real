import os
import torch
import numpy as np
from torchvision import transforms
import pandas as pd

import sys
sys.path.insert(1, './')

from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from utils import set_seed
from torchvision import transforms as T
from torch.utils.data import WeightedRandomSampler

from trainer import TrainerBase
from datasets.celeba import CelebA
from datasets.celeba_gen import CelebAGen
from datasets.utk_face import UtkFace
from datasets.utk_face_gen import UtkFaceGen
from datasets.spuco_animals import SpucoAnimals
from datasets.spuco_animals_gen import SpucoAnimalsGen

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
        self.dataset_val = NAME_TO_DATASET[opt.dataset](root_dir, split=valid_name, transform=transform["eval"], opt=opt)
        self.dataset_test = NAME_TO_DATASET[opt.dataset](root_dir, split='test', transform=transform["eval"], opt=opt)

        weights = self.dataset_train.group_counts_dro()/np.sum(self.dataset_train.group_counts_dro())
        weights = 1/weights
        weights = weights[self.dataset_train.group_idx]
        sampler = WeightedRandomSampler(np.squeeze(weights), len(weights), replacement=True)

        self.dataloader_train = DataLoader(self.dataset_train, batch_size=opt.batch_size, sampler=sampler, num_workers=2)
        self.dataloader_val = DataLoader(self.dataset_val, batch_size=opt.batch_size, num_workers=2)
        self.dataloader_test = DataLoader(self.dataset_test, batch_size=opt.batch_size, num_workers=2)

        self.dataset_train.get_class_distribution()
        self.dataset_val.get_class_distribution()
        self.dataset_test.get_class_distribution()

        gen_dir = "ratio_balance"
        opt.method = 'resample_ffr_2'

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

    def train(self):
        
        if self.opt.freeze_bn:
            self.model.eval()        
        else: 
            self.model.train()

        for epoch in range(self.epochs):
            for i, data in enumerate(tqdm(self.dataloader_train, ascii=True)):
                self.optimizer.zero_grad()
                img, label = data['img'].to('cuda'), data['target'].to('cuda')
                output = self.model(img)  

                loss = self.loss(output, label)
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
    parser.add_argument('--freeze_bn', action='store_true')

    opt = parser.parse_args()

    #define the trainer
    root_dir = get_datasets_dir(opt.dataset)

    
    trainer = ResNet50Trainer(root_dir=root_dir, transform=transform, opt=opt)
    #load the model
    trainer.load_model()
    #train the model
    trainer.train()
    trainer.test()
