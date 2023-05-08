import numpy as np
import torch


class MolecularDataset(torch.utils.data.Dataset):
    def __init__(self, adj_all, features_all, labels_all, index, max_node_num):

        self.max_num_nodes = max_node_num
        self.adj_all = adj_all
        self.len_all = []  # num of nodes (original, not padded)
        self.index = index
        self.feature_all = features_all
        self.labels_all = labels_all
        for adj in adj_all:
            self.len_all.append(adj.shape[0])

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj = self.adj_all[idx]
        return {'adj': adj.copy(),
                'features': self.feature_all[idx].copy(),
                'labels': self.labels_all[idx].copy(),
                'num_node_real': self.len_all[idx],
                'index': self.index[idx]
                }