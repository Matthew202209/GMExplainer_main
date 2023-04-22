import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np
import os, sys
import pdb
import copy
import random

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import DenseGCNConv, DenseGraphConv

from utils.Optimiser import creat_optimizer


class GANGC(object):
    def __init__(self, config, generator, discriminator, pretrain_model=None):
        # 基本参数
        self.encoder_type = config.encoder_type
        self.encode_graph_pool_type = config.encode_graph_pool_type
        self.encode_x_dim = config.encode_x_dim
        self.encode_h_dim = config.encode_h_dim

        self.is_pretrained = False
        self.is_trained = False
        # 训练参数
        # 测试参数
        # 模型
        self.D = discriminator
        self.G = generator
        self.P = pretrain_model


class Discriminator(nn.Module):
    def __int__(self, config):
        super().__int__()

        # 参数设置
        self.encode_x_dim = config.encode_x_dim
        self.encode_h_dim = config.encode_h_dim
        self.encoder_type = config.encode_type
        self.graph_pool_type = config.d_graph_pool_type
        self.dropout = config.d_dropout

        # 创建损失函数
        self.loss_function = nn.MSELoss()

        # 创建优化器
        self.optimiser = creat_optimizer(self.parameters(), lr=config.lr)

        # 计数器和进程记录
        self.counter = 0
        self.progress = []

        if self.encoder_type == 'gcn':
            self.graph_model = DenseGCNConv(self.x_dim, self.h_dim)
        elif self.encoder_type == 'graphConv':
            self.graph_model = DenseGraphConv(self.x_dim, self.h_dim)

        self.linear_layer = nn.Sequential(
            nn.Linear(self.h_dim, 64),
            nn.BatchNorm1d(self.h_dim),
            nn.Dropout(p=self.dropout, inplace=True),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, adj, features):
        graph_rep = self.graph_model(features, adj)  # n x num_node x h_dim
        graph_rep = self.graph_pooling(graph_rep, self.graph_pool_type)
        output = self.linear_layer(graph_rep)
        return output


class Generator(nn.Module):
    def __int__(self, config, vertexes):
        super().__int__()
        # 模型参数
        self.encode_x_dim = config.encode_x_dim
        self.encode_h_dim = config.encode_h_dim
        self.encoder_type = config.encode_type
        self.z_dim = config.z_dim
        self.batch_size = config.batch_size
        self.conv_dims = config.conv_dims
        self.vertexes = vertexes
        self.graph_pool_type = config.d_graph_pool_type
        self.dropout = config.d_dropout
        self.method = config.post_method

        # 创建损失函数
        self.loss_function = nn.MSELoss()

        # 创建优化器
        self.optimiser = creat_optimizer(self.parameters(), opt=config.opt, lr=config.lr)

        # 计数器和进程记录
        self.counter = 0
        self.progress = []

        if self.encoder_type == 'gcn':
            self.graph_model = DenseGCNConv(self.x_dim, self.h_dim)
        elif self.encoder_type == 'graphConv':
            self.graph_model = DenseGraphConv(self.x_dim, self.h_dim)

        layers = []
        for c0, c1 in zip([self.z_dim + self.h_dim] + self.conv_dims[:-1], self.conv_dims):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=self.dropout, inplace=True))
        self.layers = nn.Sequential(*layers)

        self.edges_layer = nn.Linear(self.conv_dims[-1], self.vertexes * self.vertexes)
        self.dropoout = nn.Dropout(p=self.dropout)

    def forward(self,  adj, features):
        z = self.sample_z(self.batch_size)
        graph_rep = self.graph_model(features, adj)  # n x num_node x h_dim
        graph_rep = self.graph_pooling(graph_rep, self.graph_pool_type)
        in_put = torch.cat([z, graph_rep], dim=1)
        out_put = self.layers(in_put)
        edges_logits = self.dropoout(self.edges_layer(out_put).view(-1, self.vertexes, self.vertexes))
        return Generator.postprocess(edges_logits, method=self.method)

    def sample_z(self, batch_size):
        return np.random.normal(0, 1, size=(batch_size, self.z_dim))

    @staticmethod
    def postprocess(inputs, method='hard_gumbel', temperature=1.):

        def listify(x):
            return x if type(x) == list or type(x) == tuple else [x]

        def delistify(x):
            return x if len(x) > 1 else x[0]

        if method == 'soft_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))
                                        / temperature, hard=False).view(e_logits.size())
                       for e_logits in listify(inputs)]
        elif method == 'hard_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))
                                        / temperature, hard=True).view(e_logits.size())
                       for e_logits in listify(inputs)]
        else:
            softmax = [F.softmax(e_logits / temperature, -1)
                       for e_logits in listify(inputs)]

        return [delistify(e) for e in softmax]


def graph_pooling(x, _type='mean'):
    global out
    if _type == 'max':
        out, _ = torch.max(x, dim=1, keepdim=False)
    elif _type == 'sum':
        out = torch.sum(x, dim=1, keepdim=False)
    elif _type == 'mean':
        out = torch.sum(x, dim=1, keepdim=False)
    return out
