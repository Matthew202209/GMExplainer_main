import time
import numpy as np
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import math
import argparse
import os
import sys
import scipy.io as scio

import numpy as np
from tqdm import tqdm

# import data_preprocessing as dpp
# import models
import utils
import random
from operator import itemgetter
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from config import PredictionModelConfig, init_prediction_args
from dataset.CausalDataset import CausalDataset
from dataset.MolecularDataset import MolecularDataset
from predictionModels.modelPool.GraphsPrediction import GraphPredModel, MolecularClassifier
from utils.LoadData import get_data_path, load_data, select_dataloader, create_experiment, select_molecular_dataloader


class GraphPredictionTrain:
    def __init__(self, args):

        self.args = args
        self.device = None
        self.pre_model = None

        # 数据集
        self.test_loader = None
        self.train_loader = None
        self.val_loader = None
        self.num_features = None

        # Prediction Model
        self.pre_model = None
        self.optimizer = None
        # initial setting
        self.set_device()
        self.get_num_features()

    def set_device(self):
        if args.cuda and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        print('using device: ', self.device)

    def set_seed(self):
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if args.cuda:
            torch.cuda.manual_seed(self.args.seed)

    def set_model(self):
        self.set_seed()
        if self.args.pre_model == 'MolecularClassifier':
            self.pre_model = MolecularClassifier(
                x_dim=self.num_features,
                h_dim=self.args.h_dim,
                num_class=self.args.num_classes,
                dropout=self.args.dropout
            ).to(self.device)
        elif self.args.pre_model == 'GraphPredModel':
            self.pre_model = GraphPredModel(
                x_dim=self.num_features,
                h_dim=self.args.h_dim,
                num_class=self.args.num_classes,
                device=self.device,
                dataset=self.args.dataset_name
            ).to(self.device)

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.pre_model.parameters(),
                                          lr=self.args.lr,
                                          weight_decay=self.args.weight_decay)

    def get_num_features(self):
        _data = load_data(get_data_path(self.args))
        num_data = _data['features_list'][0].shape[1]
        self.num_features = num_data

    def create_data_loader(self):
        # 读入数据
        global Dataset
        _data = load_data(get_data_path(self.args))
        num_data = len(_data['adj_list'])
        # 分割数据集
        self.set_seed()
        idx_train_list, idx_val_list, idx_test_list = create_experiment(1, num_data)
        train_idx = idx_train_list[0]
        val_idx = idx_val_list[0]
        test_idx = idx_test_list[0]

        if self.args.dataset_name == 'Tox21_ahr':
            self.train_loader = select_molecular_dataloader(_data, train_idx, MolecularDataset,
                                                            batch_size=args.batch_size,
                                                            num_workers=0)
            self.val_loader = select_molecular_dataloader(_data, val_idx, MolecularDataset, batch_size=args.batch_size,
                                                          num_workers=0)
            self.test_loader = select_molecular_dataloader(_data, test_idx, MolecularDataset,
                                                           batch_size=args.batch_size,
                                                           num_workers=0)
        elif self.args.dataset_name == 'ogng_molhiv' or self.args.dataset_name == 'imdb_m':
            self.train_loader = select_dataloader(_data, train_idx, CausalDataset, batch_size=args.batch_size,
                                                  num_workers=0)
            self.val_loader = select_dataloader(_data, val_idx, CausalDataset, batch_size=args.batch_size,
                                                num_workers=0)
            self.test_loader = select_dataloader(_data, test_idx, CausalDataset, batch_size=args.batch_size,
                                                 num_workers=0)

    def train(self):
        print("start training!")

        time_begin = time.time()
        best_loss = float('inf')

        for epoch in tqdm(range(self.args.epochs + 1)):
            self.pre_model.train()
            loss = 0.0
            batch_num = 0
            for batch_idx, data in enumerate(self.train_loader):
                batch_num += 1
                features = data['features'].float().to(self.device)
                adj = data['adj'].float().to(self.device)
                labels = data['labels'].float().to(self.device)
                # forward pass
                self.optimizer.zero_grad()
                model_return = self.pre_model(features, adj)
                # compute loss
                loss_params = {'model': self.pre_model, 'labels': labels}
                loss_params.update(model_return)
                loss_results = GraphPredictionTrain.compute_loss(loss_params)
                loss = loss_results['loss']
                loss.backward()
                self.optimizer.step()
            # evaluate
            if epoch % 100 == 0:
                self.pre_model.eval()
                eval_results_val = self.test(mode='val')
                eval_results_tst = self.test(mode='test')
                val_loss = eval_results_val['loss']

                metrics_results_val = ""
                metrics_results_tst = ""
                for k in self.args.metrics:
                    metrics_results_val += f"{k}_val: {eval_results_val[k]:.4f} | "
                    metrics_results_tst += f"{k}_tst: {eval_results_tst[k]:.4f} | "

                print(f"[Train] Epoch {epoch}: train_loss: {(loss):.4f} |  "
                      f"val_loss: {(val_loss):.4f} |" + metrics_results_val + metrics_results_tst +
                      f"time: {(time.time() - time_begin):.4f} |")

                # save
                if self.args.save_model:
                    val_loss = eval_results_val['loss']
                    if val_loss < best_loss:
                        best_loss = val_loss
                        self.save_model()

    def test(self, mode='val'):
        global data_loader
        if mode == 'val':
            data_loader = self.val_loader
        elif mode == 'test':
            data_loader = self.test_loader

        self.pre_model.eval()
        eval_results_all = {k: 0.0 for k in self.args.metrics}
        size_all = 0
        loss = 0.0
        batch_num = 0
        for batch_idx, data in enumerate(data_loader):
            batch_num += 1
            batch_size = len(data['labels'])
            size_all += batch_size

            features = data['features'].float().to(self.device)
            adj = data['adj'].float().to(self.device)
            labels = data['labels'].float().to(self.device)

            if (labels == 1).sum() == len(labels):
                print('no way!')

            with torch.no_grad():
                model_return = self.pre_model(features, adj)
            y_pred = model_return['y_pred']
            y_p = y_pred.argmax(dim=1).view(-1, 1)
            # compute loss
            loss_params = {'model': self.pre_model, 'labels': labels}
            loss_params.update(model_return)

            loss_results = GraphPredictionTrain.compute_loss(loss_params)
            loss_batch = loss_results['loss']
            loss += loss_batch
            # evaluate metrics
            eval_params = model_return.copy()
            eval_params.update({'y_true': labels, 'y_pred': y_pred, 'metrics': self.args.metrics})

            eval_results = GraphPredictionTrain.evaluate(eval_params)
            # batch -> all, if the metrics is calculated by averaging over all instances
            for k in self.args.metrics:
                eval_results_all[k] += (batch_size * eval_results[k])

        for k in self.args.metrics:
            eval_results_all[k] /= size_all

        loss = loss / batch_num
        eval_results_all['loss'] = loss
        return eval_results_all

    def save_model(self):
        path_model = f'D:\ProjectCodes\GMExplainer\models_save/prediction/weights_graphPred__{self.args.dataset_name}.pt'
        torch.save(self.pre_model.state_dict(), path_model)
        print('model saved in ', path_model)

    @staticmethod
    def compute_loss(params):
        labels, y_pred = params['labels'], params['y_pred']

        loss = F.nll_loss(F.log_softmax(y_pred, dim=-1), labels.view(-1).long())

        loss_results = {'loss': loss}
        return loss_results

    @staticmethod
    def evaluate(params):
        y_true, y_pred, metrics = params['y_true'], params['y_pred'], params['metrics']
        # y_pred_binary = torch.where(y_pred >= 0.5, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))
        y_pred = F.softmax(y_pred, dim=-1)
        pred_label = y_pred.argmax(dim=1).view(-1, 1)  # n x 1

        eval_results = {}
        if 'Accuracy' in metrics:
            acc = accuracy_score(y_true.cpu().numpy(), pred_label.cpu().numpy())
            eval_results['Accuracy'] = acc
        if 'AUC-ROC' in metrics:
            auc_roc = roc_auc_score(y_true.cpu().numpy(), y_pred[:, 1].detach().cpu().numpy())
            eval_results['AUC-ROC'] = auc_roc
        if 'F1-score' in metrics:
            f1 = f1_score(y_true.cpu().numpy(), pred_label.cpu().numpy())
            eval_results['F1-score'] = f1
        return eval_results

    def run(self):
        self.create_data_loader()
        self.set_model()
        self.set_optimizer()
        self.train()


def set_config(args, config):
    args.cuda = config['cuda']
    args.data_path = config['data_path']
    args.task_type = config['task_type']
    args.data_type = config['data_type']
    args.dataset_name = config['dataset_name']
    args.models_save_path = config['models_save_path']
    args.exp_type = config['exp_type']
    args.pre_model = config['model']
    args.batch_size = config['batch_size']
    args.dim_h = config['dim_h']
    args.num_class = config['num_class']
    args.epochs = config['epochs']
    args.lr = config['lr']
    args.weight_decay = config['weight_decay']
    args.save_model = config['save_model']


if __name__ == '__main__':
    config = PredictionModelConfig.Ogng_classifier_config
    args = init_prediction_args()
    set_config(args, config)
    gpt = GraphPredictionTrain(args)
    gpt.run()
