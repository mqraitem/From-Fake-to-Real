import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import pandas as pd

import sys
sys.path.insert(1, './')

from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from utils import set_seed, compute_stats
from torchvision import transforms as T
from utils import ResNet50

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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


C_OPTIONS = [3., 1., 0.3, 0.1, 0.03, 0.01, 0.003]
REG = "l1"

class ResNet50Trainer(TrainerBase):    
    def __init__(self, root_dir, transform=None, opt=None):
        self.root_dir = root_dir
        self.transform = transform
        self.opt = opt
        self.batch_size = opt.batch_size

        self.epochs = opt.epochs
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
        

        gen_dir = 'ratio_balance'
        opt.method = 'dfr_ffr_2'

        pretrain_dir =  f'{opt.dataset}/{opt.minority_to_keep}/erm_ffr_2/{gen_dir}'
        pretrain_dir_model = f"output/models/{pretrain_dir}"
        pretrain_dir_results = f"output/results/{pretrain_dir}"

        best_hyper = self.get_best_hyper(pretrain_dir_results)
        opt.pretrain_dir = f"{pretrain_dir_model}/{best_hyper}"

        self.results_dir = f'output/results/{opt.dataset}/{opt.minority_to_keep}/{opt.method}/{gen_dir}/{0.0}_{0.0}/'
        os.makedirs(self.results_dir, exist_ok=True)

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
        self.model.to('cuda')

        if self.opt.pretrain_dir != 'None': 
            print(f'Loading pretrained model from {self.opt.pretrain_dir}')
            self.model.load_state_dict(torch.load(f"{self.opt.pretrain_dir}/best_model_{self.opt.seed}.pth"))
            self.model.to('cuda')

        self.best_accuracy = -np.inf

    def get_data(self):
        self.model.eval()

        all_feat = {}
        all_label = {} 
        all_group_idx = {}
        all_bias = {}

        with torch.no_grad():

            for loader_name, data_loader in zip(["train", "val", "test"], [self.dataloader_train, self.dataloader_val, self.dataloader_test]):

                
                loader_feat = [] 
                loader_label = [] 
                loader_group_idx = [] 
                loader_bias = [] 

                for i, data in enumerate(tqdm(data_loader, ascii=True)):
                    img, label, group_idx, bias = data['img'].to('cuda'), data['target'], data['group_idx'], data["bias"]
                    
                    _, feat = self.model(img, True)  

                    loader_feat.append(feat.detach().cpu().numpy())
                    loader_label.append(label.numpy())
                    loader_group_idx.append(group_idx.numpy())
                    loader_bias.append(bias.numpy())

                loader_feat = np.concatenate(loader_feat, axis=0)
                loader_label = np.concatenate(loader_label, axis=0)
                loader_group_idx = np.concatenate(loader_group_idx, axis=0)
                loader_bias = np.concatenate(loader_bias, axis=0)

                all_feat[loader_name] = loader_feat
                all_label[loader_name] = loader_label   
                all_group_idx[loader_name] = loader_group_idx
                all_bias[loader_name] = loader_bias


        return all_feat, all_label, all_group_idx, all_bias


    def tune_dfr(self, all_feat, all_label, all_group_idx, num_retrains=1, add_train=False, balance_val=True, preprocess=True): 

        worst_accs = {}
        for i in range(num_retrains):
            x_val = all_feat["val"]
            y_val = all_label["val"]
            g_val = all_group_idx["val"]
            n_groups = np.max(g_val) + 1

            n_val = len(x_val) // 2
            idx = np.arange(len(x_val))
            np.random.shuffle(idx)

            x_valtrain = x_val[idx[n_val:]]
            y_valtrain = y_val[idx[n_val:]]
            g_valtrain = g_val[idx[n_val:]]

            n_groups = np.max(g_valtrain) + 1
            g_idx = [np.where(g_valtrain == g)[0] for g in range(n_groups)]
            min_g = np.min([len(g) for g in g_idx])
            for g in g_idx:
                np.random.shuffle(g)
            if balance_val:
                x_valtrain = np.concatenate([x_valtrain[g[:min_g]] for g in g_idx])
                y_valtrain = np.concatenate([y_valtrain[g[:min_g]] for g in g_idx])
                g_valtrain = np.concatenate([g_valtrain[g[:min_g]] for g in g_idx])

            x_val = x_val[idx[:n_val]]
            y_val = y_val[idx[:n_val]]
            g_val = g_val[idx[:n_val]]

            n_train = len(x_valtrain) if add_train else 0

            x_train = np.concatenate([all_feat["train"][:n_train], x_valtrain])
            y_train = np.concatenate([all_label["train"][:n_train], y_valtrain])
            g_train = np.concatenate([all_group_idx["train"][:n_train], g_valtrain])
            print(np.bincount(g_train))
            if preprocess:
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_val = scaler.transform(x_val)

            if balance_val and not add_train:
                cls_w_options = [{0: 1., 1: 1.}]
            else:
                cls_w_options = CLASS_WEIGHT_OPTIONS
            for c in C_OPTIONS:
                for class_weight in cls_w_options:
                    logreg = LogisticRegression(penalty=REG, C=c, 
                                                class_weight=class_weight,
                                                max_iter=10000,
                                                solver="liblinear")
                    logreg.fit(x_train, y_train)
                    preds_val = logreg.predict(x_val)
                    group_accs = np.array(
                        [(preds_val == y_val)[g_val == g].mean()
                        for g in range(n_groups)])
                    worst_acc = np.min(group_accs)
                    if i == 0:
                        worst_accs[c, class_weight[0], class_weight[1]] = worst_acc
                    else:
                        worst_accs[c, class_weight[0], class_weight[1]] += worst_acc
                    # print(c, class_weight[0], class_weight[1], worst_acc, worst_accs[c, class_weight[0], class_weight[1]])
        ks, vs = list(worst_accs.keys()), list(worst_accs.values())
        best_hypers = ks[np.argmax(vs)]
        return best_hypers

    def dfr_eval(self, c, w1, w2, all_feat, all_label, all_group_idx, all_bias, num_retrains=20, balance_val=True, preprocess=True, add_train=False ):

        coefs, intercepts = [], []
        if preprocess:
            scaler = StandardScaler()
            scaler.fit(all_feat["train"])

        for i in range(num_retrains):
            x_val = all_feat["val"]
            y_val = all_label["val"]
            g_val = all_group_idx["val"]
            n_groups = np.max(g_val) + 1
            g_idx = [np.where(g_val == g)[0] for g in range(n_groups)]
            min_g = np.min([len(g) for g in g_idx])
            for g in g_idx:
                np.random.shuffle(g)
            if balance_val:
                x_val = np.concatenate([x_val[g[:min_g]] for g in g_idx])
                y_val = np.concatenate([y_val[g[:min_g]] for g in g_idx])
                g_val = np.concatenate([g_val[g[:min_g]] for g in g_idx])

            n_train = len(x_val) if add_train else 0
            train_idx = np.arange(len(all_feat["train"]))
            np.random.shuffle(train_idx)
            train_idx = train_idx[:n_train]

            x_train = np.concatenate(
                [all_feat["train"][train_idx], x_val])
            y_train = np.concatenate([all_label["train"][train_idx], y_val])
            g_train = np.concatenate([all_group_idx["train"][train_idx], g_val])
            print(np.bincount(g_train))
            if preprocess:
                x_train = scaler.transform(x_train)

            logreg = LogisticRegression(penalty=REG, C=c, 
                                        class_weight={0: w1, 1: w2},
                                        max_iter=10000, 
                                        solver="liblinear")
            logreg.fit(x_train, y_train)
            coefs.append(logreg.coef_)
            intercepts.append(logreg.intercept_)

        x_test = all_feat["test"]
        y_test = all_label["test"]
        g_test = all_group_idx["test"]
        b_test = all_bias["test"]
        print(np.bincount(g_test))

        if preprocess:
            x_test = scaler.transform(x_test)
        logreg = LogisticRegression(penalty=REG, C=c,
                                    class_weight={0: w1, 1: w2},
                                    max_iter=10000, 
                                    solver="liblinear")
        n_classes = np.max(y_train) + 1
        # the fit is only needed to set up logreg
        logreg.fit(x_train[:n_classes], np.arange(n_classes))
        logreg.coef_ = np.mean(coefs, axis=0)
        logreg.intercept_ = np.mean(intercepts, axis=0)
            
        preds_test = logreg.predict(x_test)

        x_val = all_feat["val"]
        y_val = all_label["val"]
        g_val = all_group_idx["val"]
        b_val = all_bias["val"]

        x_val = scaler.transform(x_val)
        preds_val = logreg.predict(x_val)

        worst_acc_val, balanced_acc_val = compute_stats(preds_val, y_val, b_val, self.dataset_train.GROUPS, take_argmax=False)
        worst_acc_test, balanced_acc_test  = compute_stats(preds_test, y_test, b_test, self.dataset_train.GROUPS, take_argmax=False)

        results = { 
            "worst_acc_val": worst_acc_val,
            "balanced_acc_val": balanced_acc_val,
            "worst_acc_test": worst_acc_test,
            "balanced_acc_test": balanced_acc_test,
        }

        print(results)


        df = pd.DataFrame(results, index=[0])
        df.to_csv(os.path.join(self.results_dir, f'results_{self.opt.seed}.csv'))

    def get_results(self): 
        all_feat, all_label, all_group_idx, all_bias = self.get_data()
        c, w1, w2 = self.tune_dfr(all_feat, all_label, all_group_idx, num_retrains=1)
        self.dfr_eval(c, w1, w2, all_feat, all_label, all_group_idx, all_bias, num_retrains=20)


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
    parser.add_argument('--minority_to_keep', type=float, default=0.95)
    opt = parser.parse_args()
    root_dir = get_datasets_dir(opt.dataset)

    
    trainer = ResNet50Trainer(root_dir=root_dir, transform=transform, opt=opt)
    #load the model
    trainer.load_model()
    #train the model
    trainer.get_results()
