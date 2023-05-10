import pickle
import random

import numpy as np
import torch
from operator import itemgetter

from config import init_args_gcf_gan


def load_data(path):
    f = open(path, 'rb')
    data = pickle.load(f)
    f.close()
    return data

def get_data_path(args):
    return r'{}/{}/{}/{}.pkl'.format(args.data_path, args.task_type, args.data_type, args.dataset_name)

def get_prediction_models_save_path(args):
    return r'{}/{}.pt'.format(args.models_save_path, args.prediction)
def get_items_from_list(li, idx_select):
    items = itemgetter(*idx_select)(li)
    return items


def select_dataloader(data, idx_select, DataSet, batch_size=500, num_workers=0, padded=False):
    adj = get_items_from_list(data['adj_list'],idx_select)
    features = get_items_from_list(data['features_list'], idx_select)
    u = get_items_from_list(data['u_all'], idx_select)
    labels = get_items_from_list(data['labels'], idx_select)
    num_node_real = get_items_from_list(data['num_node_real'], idx_select)
    max_num_node = data['max_num_node']
    dataset_select = DataSet(adj, features,
                             u, labels, num_node_real, max_num_node ,idx_select,
                             padded)
    data_loader_select = torch.utils.data.DataLoader(
        dataset_select,
        batch_size=batch_size,
        num_workers=num_workers,drop_last=True)
    return data_loader_select


def select_molecular_dataloader(data, idx_select, DataSet, batch_size=500, num_workers=0):
    adj = get_items_from_list(data['adj_list'], idx_select)
    features = get_items_from_list(data['features_list'], idx_select)
    labels = get_items_from_list(data['label_list'], idx_select)
    max_node_num = data['max_node_num']
    dataset_select = DataSet(adj, features,
                             labels, idx_select,
                             max_node_num)
    data_loader_select = torch.utils.data.DataLoader(
        dataset_select,
        batch_size=batch_size,
        num_workers=num_workers, drop_last=True)
    return data_loader_select


def create_dataset_index(num_data):
    list_index = [x for x in range(num_data)]
    random.shuffle(list_index)
    train_index = list_index[0:int(num_data * 0.8) - 1]
    val_index = list_index[int(num_data * 0.8): int(num_data * 0.9) - 1]
    test_index = list_index[int(num_data * 0.9):]
    return train_index, val_index, test_index


def create_experiment(num_expr, num_data):
    idx_train_list, idx_val_list, idx_test_list = [], [], []
    for i in range(num_expr):
        train_index, val_index, test_index = create_dataset_index(num_data)
        idx_train_list.append(train_index)
        idx_val_list.append(val_index)
        idx_test_list.append(test_index)
    return idx_train_list, idx_val_list, idx_test_list


def save_experiment(num_expr, num_data, path):
    expr_dict = {}
    idx_train_list, idx_val_list, idx_test_list = create_experiment(num_expr, num_data)
    expr_dict['idx_train_list'] = idx_train_list
    expr_dict['idx_val_list'] = idx_val_list
    expr_dict['idx_test_list'] = idx_test_list
    f = open(path, 'wb')
    pickle.dump(expr_dict, f)


if __name__ == '__main__':
    args = init_args_gcf_gan()
    data_path = get_data_path(args)
    data = load_data(data_path)
    num_data = len(data['adj_list'])

    save_dir = 'D:\ProjectCodes\GMExplainer\data\experiment'
    save_file = r'expr_{}.pkl'.format(args.dataset_name)
    save_path = r'{}/{}'.format(save_dir, save_file)
    save_experiment(10, num_data, save_path)


