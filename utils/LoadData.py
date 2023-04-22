import pickle

import numpy as np
import torch
from operator import itemgetter

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
def select_dataloader(data, idx_select, DataSet, batch_size=500, num_workers=0, padded = False):
    adj = get_items_from_list(data['adj_list'],idx_select)
    features = get_items_from_list(data['features_list'], idx_select)
    u = get_items_from_list(data['u_all'], idx_select)
    labels = get_items_from_list(data['labels'], idx_select)

    dataset_select = DataSet(adj, features,
                             u, labels,idx_select,
                             padded)
    data_loader_select = torch.utils.data.DataLoader(
        dataset_select,
        batch_size=batch_size,
        num_workers=num_workers)
    return data_loader_select