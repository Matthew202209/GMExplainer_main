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
# import data_preprocessing as dpp
# import models
import utils
import random
from operator import itemgetter
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from config import get_args
from dataset.CausalDataset import CausalDataset
from dataset.MolecularDataset import MolecularDataset
from predictionModels.modelPool.GraphsPrediction import Graph_pred_model
from utils.LoadData import get_data_path, load_data, select_dataloader, create_experiment, select_molecular_dataloader

sys.path.append('../')
torch.backends.cudnn.enabled = False


parser = argparse.ArgumentParser(description='Graph counterfactual explanation generation')
parser.add_argument('--nocuda', type=int, default=0, help='Disables CUDA training.')
parser.add_argument('--batch_size', type=int, default=5000, metavar='N',
                    help='input batch size for training (default: 500)')  # community: 500， ogbg: 5000
parser.add_argument('--num_workers', type=int, default=0, metavar='N')
parser.add_argument('--epochs', type=int, default=600, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--trn_rate', type=float, default=0.6, help='training data ratio')
parser.add_argument('--tst_rate', type=float, default=0.2, help='test data ratio')

parser.add_argument('--dim_h', type=int, default=32, metavar='N', help='dimension of h')
parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('--dataset', default='ogbg_molhiv', help='dataset to use',
                    choices=['community', 'ogbg_molhiv', 'imdb_m'])
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate for optimizer')  # community: 1e-3
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='weight decay')

args = parser.parse_args()

# select gpu if available
if args.cuda and torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print('using device: ', device)

# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def compute_loss(params):
    labels, y_pred = params['labels'], params['y_pred']

    loss = F.nll_loss(F.log_softmax(y_pred, dim=-1), labels.view(-1).long())

    loss_results = {'loss': loss}
    return loss_results


def train(params):
    epochs, model, optimizer, train_loader, val_loader, test_loader, dataset, metrics = \
        params['epochs'], params['model'], params['optimizer'], \
        params['train_loader'], params['val_loader'], params['test_loader'], params['dataset'], params['metrics']
    save_model = params['save_model'] if 'save_model' in params else True
    print("start training!")

    time_begin = time.time()
    best_loss = float('inf')

    for epoch in range(epochs + 1):
        model.train()

        loss = 0.0
        batch_num = 0
        for batch_idx, data in enumerate(train_loader):
            batch_num += 1
            model.zero_grad()

            features = data['features'].float().to(device)
            adj = data['adj'].float().to(device)
            labels = data['labels'].float().to(device)

            optimizer.zero_grad()

            # forward pass
            model_return = model(features, adj)

            # compute loss
            loss_params = {'model': model, 'labels': labels}
            loss_params.update(model_return)

            loss_results = compute_loss(loss_params)
            loss_batch = loss_results['loss']
            loss += loss_batch

        # backward propagation
        (loss / batch_num).backward()
        optimizer.step()

        # evaluate
        if epoch % 100 == 0:
            model.eval()
            eval_params_val = {'model': model, 'data_loader': val_loader, 'dataset': dataset, 'metrics': metrics}
            eval_params_tst = {'model': model, 'data_loader': test_loader, 'dataset': dataset, 'metrics': metrics}
            eval_results_val = test(eval_params_val)
            eval_results_tst = test(eval_params_tst)
            val_loss = eval_results_val['loss']

            metrics_results_val = ""
            metrics_results_tst = ""
            for k in metrics:
                metrics_results_val += f"{k}_val: {eval_results_val[k]:.4f} | "
                metrics_results_tst += f"{k}_tst: {eval_results_tst[k]:.4f} | "

            print(f"[Train] Epoch {epoch}: train_loss: {(loss):.4f} |  "
                  f"val_loss: {(val_loss):.4f} |" + metrics_results_val + metrics_results_tst +
                  f"time: {(time.time() - time_begin):.4f} |")

            # save
            if save_model:
                val_loss = eval_results_val['loss']
                if val_loss < best_loss:
                    best_loss = val_loss
                    path_model = f'D:\ProjectCodes\GMExplainer\models_save/prediction/weights_graphPred__{args.dataset_name}.pt'
                    torch.save(model.state_dict(), path_model)
                    print('model saved in ', path_model)

    return


def test(params):
    model, data_loader, dataset, metrics = params['model'], params['data_loader'], params['dataset'], params['metrics']
    model.eval()

    eval_results_all = {k: 0.0 for k in metrics}
    size_all = 0
    loss = 0.0
    batch_num = 0
    for batch_idx, data in enumerate(data_loader):
        batch_num += 1
        batch_size = len(data['labels'])
        size_all += batch_size

        features = data['features'].float().to(device)
        adj = data['adj'].float().to(device)
        # u = data['u'].float().to(device)
        labels = data['labels'].float().to(device)
        orin_index = data['index']

        if (labels == 1).sum() == len(labels):
            print('no way!')

        with torch.no_grad():
            model_return = model(features, adj)
        y_pred = model_return['y_pred']
        y_p = y_pred.argmax(dim=1).view(-1, 1)
        acc = accuracy_score(labels.cpu().numpy(), y_p.cpu().numpy())
        # compute loss
        loss_params = {'model': model, 'labels': labels}
        loss_params.update(model_return)

        loss_results = compute_loss(loss_params)
        loss_batch = loss_results['loss']
        loss += loss_batch

        # evaluate metrics
        eval_params = model_return.copy()
        eval_params.update({'y_true': labels, 'y_pred': y_pred, 'metrics': metrics})

        eval_results = evaluate(eval_params)
        # batch -> all, if the metrics is calculated by averaging over all instances
        for k in metrics:
            eval_results_all[k] += (batch_size * eval_results[k])

    for k in metrics:
        eval_results_all[k] /= size_all

    loss = loss / batch_num
    eval_results_all['loss'] = loss
    return eval_results_all

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
        auc_roc = roc_auc_score(y_true.cpu().numpy(), y_pred[:,1].detach().cpu().numpy())
        eval_results['AUC-ROC'] = auc_roc
    if 'F1-score' in metrics:
        f1 = f1_score(y_true.cpu().numpy(), pred_label.cpu().numpy())
        eval_results['F1-score'] = f1
    return eval_results


if __name__ == '__main__':
    args = get_args()
    data_path = get_data_path(args)
    data = load_data(data_path)
    num_data = len(data['adj_list'])
    idx_train_list, idx_val_list, idx_test_list = create_experiment(1, num_data)
    train_idx = idx_train_list[0]
    val_idx = idx_val_list[0]
    test_idx = idx_test_list[0]


    # train_idx = data['train_idx']
    # val_idx = data['val_idex']
    # test_idx = data['test_idx']
    #
    # train_node_num = data['train_node_num']
    # val_node_num = data['val_node_num']
    # test_node_num = data['test_node_num']
    #
    train_loader = select_molecular_dataloader(data, train_idx, MolecularDataset, batch_size=args.batch_size,
                                     num_workers=0)
    val_loader = select_molecular_dataloader(data, val_idx, MolecularDataset, batch_size=args.batch_size,
                                     num_workers=0)
    test_loader = select_molecular_dataloader(data, test_idx, MolecularDataset, batch_size=args.batch_size,
                                     num_workers=0)
    #
    metrics = ['Accuracy', 'AUC-ROC', 'F1-score']
    x_dim = data["features_list"][0].shape[1]
    max_num_nodes = data["adj_list"][0].shape[1]
    model = Graph_pred_model(x_dim, args.dim_h, args.num_class, max_num_nodes, args.dataset_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model = model.to(device)

    train_params = {'epochs': args.epochs, 'model': model, 'optimizer': optimizer,
                    'train_loader': train_loader, 'val_loader': val_loader, 'test_loader': test_loader,
                    'dataset': args.dataset_name, 'metrics': metrics, 'save_model': True}
    train(train_params)

    # test
    # model = models.GraphCFE(init_params=init_params, args=args).to(device)
    # model.load_state_dict(torch.load(model_path + f'weights_graphCFE_{args.dataset}_exp' + str(exp_i) + '.pt'))
    #
    # test_params = {'model': model, 'dataset': args.dataset_name, 'data_loader': test_loader, 'metrics': metrics}
    # eval_results = test(test_params)
    #
    # for k in eval_results:
    #     if isinstance(eval_results[k], list):
    #         print(k, ": ", eval_results[k])
    #     else:
    #         print(k, f": {eval_results[k]:.4f}")



