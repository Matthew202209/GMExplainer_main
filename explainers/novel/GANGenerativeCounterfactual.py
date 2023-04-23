import argparse
import time

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

from predictionModels.PredictionModelBuilder import build_prediction_model
from utils.Optimiser import creat_optimizer


class GCFGAN(object):
    def __init__(self, args):
        # 参数
        self.args = args
        # 训练设备
        self.device = self.args.device
        # 数据集
        self.data = None
        self.is_pretrained = False
        self.is_trained = False
        # 训练参数
        # 测试参数
        # 模型
        self.D = None
        self.G = None
        self.P = None
        self.pred_model = None
    """数据处理"""
    # 载入数据
    def load_data(self):
        pass
    # 创建data_loader

    def create_data_loader(self, mode='train'):
        dataloader = None
        return dataloader

    # 创建反事实标签
    def create_cf_label(self, orin_index):
        y_cf = 1 - np.array(self.data.labels_all)
        y_cf = torch.FloatTensor(y_cf).to(self.device)
        return y_cf[orin_index]

    def build_model(self):
        x_dim = self.data["features_list"][0].shape[1]
        max_num_nodes = self.data["adj_list"][0].shape[1]
        # 创建判别器和构造器
        self.D = Discriminator(self.args, x_dim)
        self.G = Generator(self.args, max_num_nodes, x_dim)

        # 打印模型结构
        GCFGAN.print_network(self.G, 'G')
        GCFGAN.print_network(self.D, 'D')

        # 设置训练设备
        self.D.to(self.device)
        self.G.to(self.device)

        self.pred_model = build_prediction_model(self.args, x_dim, max_num_nodes)

    def train(self):
        data_loader = self.create_data_loader(mode='train')
        best_loss = 100000
        print('Start training...')
        for epoch in range(self.args.epoches):
            time_begin = time.time()
            self.D.train()
            self.G.train()
            self.pred_model.eval()
            # =================================================================================== #
            #                             1. Prepare data                                         #
            # =================================================================================== #
            for i, element in enumerate(tqdm(data_loader)):
                adj, x, orin_index = element['adj'], element['features'], element['orin_index']

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #
                for d_time in range(self.args.d_train_t):
                    # 清零优化器
                    self.reset_grad()
                    # 计算真实数据的loss
                    logits_real = self.D(adj, x)
                    d_loss_real = - torch.mean(logits_real)
                    # 创造反事实例子
                    c_adj = self.G(adj, x)
                    c_adj = c_adj.detach()
                    # 计算创造数据的loss
                    logits_fake = self.D(c_adj, x)
                    d_loss_fake = torch.mean(logits_fake)
                    d_loss = d_loss_fake + d_loss_real
                    # 更新参数
                    d_loss.backward()
                    self.D.optimizer.step()

                    if d_time == self.args.d_train_t-1:
                        self.D.progress['D/loss_real'] = d_loss_real.item()
                        self.D.progress['D/loss_fake'] = d_loss_fake.item()
                        self.D.progress['D/loss_total'] = d_loss.item()

            # =================================================================================== #
            #                             3. Train the generator                                  #
            # =================================================================================== #

                for g_time in range(self.args.g_train_t):
                    self.reset_grad()
                    # 创造反事实
                    c_adj = self.G(adj, x)
                    # 计算创造数据的loss
                    logits_fake = self.D(c_adj, x)
                    g_loss_fake = -torch.mean(logits_fake)

                    # 创建反事实标签
                    y_cf = self.create_cf_label(orin_index)
                    # 反事实损失
                    y_pred = self.pred_model(x, adj)  # n x num_class
                    g_loss_cfe = F.nll_loss(F.log_softmax(y_pred, dim=-1), y_cf.view(-1).long())
                    g_loss = g_loss_fake + g_loss_cfe
                    g_loss.backward()
                    self.G.optimizer.step()

                    if g_time == self.args.d_train_t - 1:
                        self.G.progress['G/loss_fake'] = g_loss_fake.item()
                        self.G.progress['G/loss_cf'] = g_loss_cfe.item()
                        self.G.progress['G/loss_total'] = g_loss.item()

        pass

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.D.optimiser.zero_grad()
        self.G.optimiser.zero_grad()

    @staticmethod
    def print_network(model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))


class Discriminator(nn.Module):
    def __init__(self, args, x_dim):
        super(Discriminator, self).__init__()

        # 参数设置
        self.encode_x_dim = x_dim
        self.encode_h_dim = args.encode_h_dim
        self.encoder_type = args.encode_type
        self.graph_pool_type = args.d_graph_pool_type
        self.dropout = args.d_dropout

        # 计数器和进程记录
        self.counter = 0
        self.progress = {}

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
        # 创建损失函数
        self.loss_function = nn.MSELoss()
        # 创建优化器
        self.optimiser = creat_optimizer(self.parameters(), opt=args.opt, lr=args.d_lr)

    def forward(self, adj, features):
        graph_rep = self.graph_model(features, adj)  # n x num_node x h_dim
        graph_rep = self.graph_pooling(graph_rep, self.graph_pool_type)
        output = self.linear_layer(graph_rep)
        return output


class Generator(nn.Module):
    def __init__(self, args, max_num_nodes, x_dim):
        super(Generator).__init__()
        # 模型参数
        self.encode_x_dim = x_dim
        self.encode_h_dim = args.encode_h_dim
        self.encoder_type = args.encode_type
        self.z_dim = args.z_dim
        self.batch_size = args.batch_size
        self.conv_dims = args.conv_dims
        self.max_num_nodes = max_num_nodes
        self.graph_pool_type = args.d_graph_pool_type
        self.dropout = args.d_dropout
        self.post_method = args.post_method

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

        self.edges_layer = nn.Linear(self.conv_dims[-1], self.max_num_nodes * self.max_num_nodes)
        self.dropoout = nn.Dropout(p=self.dropout)

        # 创建损失函数
        self.loss_function = nn.MSELoss()

        # 创建优化器
        self.optimiser = creat_optimizer(self.parameters(), opt=args.opt, lr=args.g_lr)

    def forward(self, adj, features):
        z = self.sample_z(self.batch_size)
        graph_rep = self.graph_model(features, adj)  # n x num_node x h_dim
        graph_rep = self.graph_pooling(graph_rep, self.graph_pool_type)
        in_put = torch.cat([z, graph_rep], dim=1)
        out_put = self.layers(in_put)
        edges_logits = self.dropoout(self.edges_layer(out_put).view(-1, self.max_num_nodes, self.max_num_nodes))
        return Generator.postprocess(edges_logits, method=self.post_method)

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

if __name__ == '__main__':
    a = {}
    a['k'] = [1]
    a['k'] += [2]
    print(a)