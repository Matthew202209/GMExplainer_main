import pickle
import random

import numpy as np
from rdkit import Chem
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch.nn import functional as F
from torch_geometric.data import Data
from enum import Enum


def e_map_tox21(bond_type, reverse=False):
    if not reverse:
        if bond_type == Chem.BondType.SINGLE:
            return 0
        elif bond_type == Chem.BondType.DOUBLE:
            return 1
        elif bond_type == Chem.BondType.AROMATIC:
            return 2
        elif bond_type == Chem.BondType.TRIPLE:
            return 3
        else:
            raise Exception("No bond type found")

    if bond_type == 0:
        return Chem.BondType.SINGLE
    elif bond_type == 1:
        return Chem.BondType.DOUBLE
    elif bond_type == 2:
        return Chem.BondType.AROMATIC
    elif bond_type == 3:
        return Chem.BondType.TRIPLE
    else:
        raise Exception("No bond type found")


class x_map_tox21(Enum):
    O = 0
    C = 1
    N = 2
    F = 3
    Cl = 4
    S = 5
    Br = 6
    Si = 7
    Na = 8
    I = 9
    Hg = 10
    B = 11
    K = 12
    P = 13
    Au = 14
    Cr = 15
    Sn = 16
    Ca = 17
    Cd = 18
    Zn = 19
    V = 20
    As = 21
    Li = 22
    Cu = 23
    Co = 24
    Ag = 25
    Se = 26
    Pt = 27
    Al = 28
    Bi = 29
    Sb = 30
    Ba = 31
    Fe = 32
    H = 33
    Ti = 34
    Tl = 35
    Sr = 36
    In = 37
    Dy = 38
    Ni = 39
    Be = 40
    Mg = 41
    Nd = 42
    Pd = 43
    Mn = 44
    Zr = 45
    Pb = 46
    Yb = 47
    Mo = 48
    Ge = 49
    Ru = 50
    Eu = 51
    Sc = 52


def pyg_to_mol_tox21(pyg_mol):
    mol = Chem.RWMol()

    X = pyg_mol.x.numpy().tolist()
    X = [
        Chem.Atom(x_map_tox21(x.index(1)).name)
        for x in X
    ]

    E = pyg_mol.edge_index.t()

    for x in X:
        mol.AddAtom(x)

    for (u, v), attr in zip(E, pyg_mol.edge_attr):
        u = u.item()
        v = v.item()
        attr = attr.numpy().tolist()
        attr = e_map_tox21(attr.index(1), reverse=True)

        if mol.GetBondBetweenAtoms(u, v):
            continue

        mol.AddBond(u, v, attr)

    return mol


def pre_transform(sample, n_pad):
    sample.x = F.pad(sample.x, (0, n_pad), "constant")
    return sample


def check_molecule_validity(mol, transform):
    if type(mol) == Data:
        mol = transform(mol)

    return Chem.SanitizeMol(mol, catchErrors=True) == Chem.SANITIZE_NONE


def _preprocess_tox21():
    dataset_tr, dataset_vl, dataset_ts = load_tox21()
    data_list = (
            [dataset_tr.get(idx) for idx in range(len(dataset_tr))] +
            [dataset_vl.get(idx) for idx in range(len(dataset_vl))] +
            [dataset_ts.get(idx) for idx in range(len(dataset_ts))]
    )

    data_list = list(filter(lambda mol: check_molecule_validity(mol, pyg_to_mol_tox21), data_list))
    POSITIVES = list(filter(lambda x: x.y == 1, data_list))
    NEGATIVES = list(filter(lambda x: x.y == 0, data_list))
    N_POSITIVES = len(POSITIVES)
    N_NEGATIVES = N_POSITIVES
    NEGATIVES = NEGATIVES[:N_NEGATIVES]
    data_list = POSITIVES + NEGATIVES
    random.shuffle(data_list)
    processed_data_dict = {
        'adj_list': [],
        'edge_attr': [],
        'features_list': [],
        'label_list': [],
        'num_positives': N_POSITIVES,
        'num_negatives': N_NEGATIVES
    }
    max_node_num = -1
    for data in data_list:
        if data.stores[0]['x'].numpy().shape[0] > max_node_num:
            max_node_num = data.stores[0]['x'].numpy().shape[0]
    for data in data_list:
        edges_index = data.stores[0]['edge_index'].numpy()
        adj = edges_2_adj(edges_index, max_node_num)
        x = data.stores[0]['x'].numpy()
        features = x_2_features(x, max_node_num)
        processed_data_dict['adj_list'].append(adj)
        processed_data_dict['edge_attr'].append(data.stores[0]['edge_attr'].numpy())
        processed_data_dict['features_list'].append(features)
        processed_data_dict['label_list'].append(data.stores[0]['y'].numpy())
    processed_data_dict['max_node_num'] = max_node_num

    f = open(r'data\graph_classification\realworld\Tox21_ahr.pkl', 'wb')
    pickle.dump(processed_data_dict, f)



def load_tox21():
    dataset_tr = TUDataset('data/raw/tox21',
                           name='Tox21_AhR_training',
                           pre_transform=lambda sample: pre_transform(sample, 3))

    dataset_vl = TUDataset('data/raw/tox21',
                           name='Tox21_AhR_evaluation',
                           pre_transform=lambda sample: pre_transform(sample, 0))

    dataset_ts = TUDataset('data/raw/tox21',
                           name='Tox21_AhR_testing',
                           pre_transform=lambda sample: pre_transform(sample, 2))
    return dataset_tr, dataset_vl, dataset_ts


def edges_2_adj(edge_index, max_num_data):
    adj = np.zeros([max_num_data, max_num_data])
    adj_eye = np.eye(max_num_data)
    adj[edge_index[0], edge_index[1]] = 1
    adj = adj + adj_eye
    return adj


def x_2_features(x, max_num_data):
    i = np.array([a for a in range(x.shape[0])])
    j = np.argmax(x, axis = 1)
    features = np.zeros([max_num_data, x.shape[1]])
    features[i, j] = 1.
    return features

import pybel

if __name__ == '__main__':
    data_list = _preprocess_tox21()


    # file_path = r'D:\ProjectCodes\GMExplainer\data\graph_classification\realworld\tox21_10k_data_all.sdf'
    # with open(file_path, 'r') as f:  # 打开sdf文件
    #     lines = f.readlines()
    # print(1)
