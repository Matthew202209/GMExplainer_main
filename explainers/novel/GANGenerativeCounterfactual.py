import argparse
import time
import datetime
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

from Evaluation import evaluate
from config import get_args_for_gcf_gan
from predictionModels.PredictionModelBuilder import build_prediction_model
from utils.LoadData import get_data_path, load_data, select_dataloader
from utils.Optimiser import creat_optimizer

torch.set_default_tensor_type(torch.DoubleTensor)


class GCFGAN(object):
    def __init__(self, args, all_data):
        # 参数
        self.args = args
        # 训练设备
        self.device = self.args.device
        # 数据集
        self.all_data = all_data
        self.y_cf = None
        # 训练参数
        # 测试参数
        # 模型
        self.D = None
        self.G = None
        self.P = None
        self.pred_model = None
        # 地址
        # self.model_save_dir = self.args.model_save_dir

    """数据处理"""

    # 载入数据

    # 创建data_loader

    def create_data_loader(self, mode='train'):
        _package_ = __import__("dataset.{}".format(self.args.used_dataset), fromlist=[self.args.used_dataset])
        dataset = getattr(_package_, self.args.used_dataset)
        if mode == 'train':
            data_idx = self.all_data['train_idx']
            dataloader = select_dataloader(self.all_data, data_idx, dataset, batch_size=self.args.batch_size,
                                           num_workers=0, padded=False)
        elif mode == 'val':
            data_idx = self.all_data['val_idex']
            dataloader = select_dataloader(self.all_data, data_idx, dataset, batch_size=len(data_idx),
                                           num_workers=0, padded=False)
        elif mode == 'test':
            data_idx = self.all_data['test_idx']
            dataloader = select_dataloader(self.all_data, data_idx, dataset, batch_size=len(data_idx),
                                           num_workers=0, padded=False)
        return dataloader

    # 创建反事实标签
    def create_cf_label(self):
        y_cf = 1 - np.array(self.all_data['labels'])
        self.y_cf = torch.FloatTensor(y_cf).to(self.device)

    def select_cf_label(self, orin_index):
        return self.y_cf[orin_index]

    def build_model(self):
        x_dim = self.all_data["features_list"][0].shape[1]
        max_num_nodes = self.all_data["adj_list"][0].shape[1]
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
        self.pred_model.to(self.device)

    def train(self):
        global d_loss_real, d_loss_fake, d_loss, g_loss_fake, g_loss_cfe, g_loss
        data_loader = self.create_data_loader(mode='train')
        validity = -1
        print('Start training...')
        # 生成反事实标签
        self.create_cf_label()
        time_begin = time.time()
        for epoch in range(self.args.epoches):
            self.D.train()
            self.G.train()
            self.pred_model.eval()
            # =================================================================================== #
            #                             1. Prepare data                                         #
            # =================================================================================== #

            for i, element in enumerate(data_loader):
                adj, x, orin_index = element['adj'], element['features'], element['index']

                # 创建反事实标签
                y_cf = self.select_cf_label(orin_index)

                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #
                for d_time in range(self.args.d_train_t):
                    # 清零优化器

                    # 计算真实数据的loss
                    logits_real = self.D(adj, x)
                    d_loss_real = - torch.mean(logits_real)
                    # 创造反事实例子
                    z = self.sample_z(self.args.batch_size)
                    c_adj = self.G(adj, x, z, y_cf)
                    c_adj = c_adj.detach()
                    # 计算创造数据的loss
                    logits_fake = self.D(c_adj, x)
                    d_loss_fake = torch.mean(logits_fake)
                    d_loss = d_loss_fake + d_loss_real
                    # 更新参数
                    self.reset_grad()
                    d_loss.backward()
                    self.D.optimizer.step()

                # =================================================================================== #
                #                             3. Train the generator                                  #
                # =================================================================================== #

                for g_time in range(self.args.g_train_t):
                    # 创造反事实
                    z = self.sample_z(self.args.batch_size)
                    c_adj = self.G(adj, x, z, y_cf)
                    # 计算创造数据的loss
                    logits_fake = self.D(c_adj, x)
                    g_loss_fake = -torch.mean(logits_fake)
                    # 反事实损失
                    y_pred = self.pred_model(x, c_adj)['y_pred']  # n x num_class
                    g_loss_cfe = F.nll_loss(F.log_softmax(y_pred, dim=-1), y_cf.view(-1).long())
                    g_loss = g_loss_fake
                    self.reset_grad()
                    g_loss.backward()
                    self.G.optimizer.step()

            self.D.progress['D/loss_real'] += [d_loss_real.item()]
            self.D.progress['D/loss_fake'] += [d_loss_fake.item()]
            self.D.progress['D/loss_total'] += [d_loss.item()]
            self.G.progress['G/loss_fake'] += [g_loss_fake.item()]
            self.G.progress['G/loss_cf'] += [g_loss_cfe.item()]
            self.G.progress['G/loss_total'] += [g_loss.item()]

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            if epoch % 100 == 0:
                et = time.time() - time_begin


                loss_dict = {'epoch': epoch,
                             'D/loss_real': self.D.progress['D/loss_real'][-1],
                             'D/loss_fake': self.D.progress['D/loss_fake'][-1],
                             'D/loss_total': self.D.progress['D/loss_total'][-1],
                             'G/loss_fake': self.G.progress['G/loss_fake'][-1],
                             'G/loss_cf': self.G.progress['G/loss_cf'][-1],
                             'G/loss_total': self.G.progress['G/loss_total'][-1]}

                eval_results_all = self.evaluation()
                eval_results_all.update(loss_dict)
                print_eval_results(eval_results_all, et)
                if eval_results_all['validity'] > validity:
                    validity = eval_results_all['validity']
                    G_path = os.path.join(self.args.model_save_dir, 'best-G.ckpt')
                    D_path = os.path.join(self.args.model_save_dir, 'best-D.ckpt')
                    torch.save(self.G.state_dict(), G_path)
                    torch.save(self.D.state_dict(), D_path)
                    print('Saved model checkpoints into {}...'.format(self.args.model_save_dir))

                if (epoch + 1) % self.args.lr_update_step == 0 and (epoch + 1) > (
                        self.args.epoches - self.args.num_epoches_lr_decay):
                    self.G.g_lr -= (self.args.g_lr / float(self.args.num_epoches_lr_decay))
                    self.D.d_lr -= (self.args.d_lr / float(self.args.num_epoches_lr_decay))
                    self.update_lr()
                    print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(self.G.g_lr, self.D.d_lr))

    def evaluation(self):
        # 调整模型状态
        self.D.eval()
        self.G.eval()

        eval_results_all = {}
        data_loader = self.create_data_loader(mode='val')
        with torch.no_grad():
            for i, element in enumerate(data_loader):
                adj, x, orin_index = element['adj'], element['features'], element['index']
                y_cf = self.select_cf_label(orin_index)
                z = self.sample_z(orin_index.shape[0])
                c_adj = self.G(adj, x, z, y_cf)
                # 计算创造数据的loss
                logits_fake = self.D(c_adj, x)
                similarity = torch.mean(logits_fake).item()

                y_cf_pred = self.pred_model(x, c_adj)['y_pred']

                # 计算评价指标
                eval_params = {'pred_model': self.pred_model, 'adj_input': adj, 'x_input': x, 'adj_reconst': c_adj,
                               'y_cf': y_cf, 'y_cf_pred': y_cf_pred,
                               'metrics': self.args.metrics, 'device': self.args.device}
                eval_results = evaluate(eval_params)
                eval_results.update({'similarity': similarity})
                eval_results_all.update(eval_results)

        return eval_results_all

    def sample_z(self, batch_size):
        s_z = np.random.normal(0, 1, size=(batch_size, self.args.z_dim))
        tensor_s_z = torch.FloatTensor(s_z).to(self.device)
        return tensor_s_z

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.D.optimizer.zero_grad()
        self.G.optimizer.zero_grad()

    def update_lr(self):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.G.optimizer.param_groups:
            param_group['lr'] = self.G.g_lr
        for param_group in self.D.optimizer.param_groups:
            param_group['lr'] = self.D.d_lr

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
        self.d_lr = args.d_lr

        # 计数器和进程记录
        self.counter = 0
        self.progress = {'D/loss_real': [],
                         'D/loss_fake': [],
                         'D/loss_total': []
                         }

        if self.encoder_type == 'gcn':
            self.graph_model = DenseGCNConv(self.encode_x_dim, self.encode_h_dim)
        elif self.encoder_type == 'graphConv':
            self.graph_model = DenseGraphConv(self.encode_x_dim, self.encode_h_dim)

        self.linear_layer = nn.Sequential(
            nn.Linear(self.encode_h_dim, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(p=self.dropout, inplace=True),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        # 创建损失函数
        self.loss_function = nn.MSELoss()
        # 创建优化器
        self.optimizer = creat_optimizer(self.parameters(), opt=args.opt, lr=self.d_lr)

    def forward(self, adj, features):
        graph_rep = self.graph_model(features, adj)  # n x num_node x h_dim
        graph_rep = graph_pooling(graph_rep, self.graph_pool_type)
        output = self.linear_layer(graph_rep)
        return output


class Generator(nn.Module):
    def __init__(self, args, max_num_nodes, x_dim):
        super(Generator, self).__init__()
        # 模型参数
        self.encode_x_dim = x_dim
        self.encode_h_dim = args.encode_h_dim
        self.encoder_type = args.encode_type
        self.z_dim = args.z_dim
        self.batch_size = args.batch_size
        self.conv_dims = args.conv_dims
        self.max_num_nodes = max_num_nodes
        self.graph_pool_type = args.g_graph_pool_type
        self.dropout = args.g_dropout
        self.post_method = args.post_method
        self.g_lr = args.g_lr

        # 计数器和进程记录
        self.counter = 0
        self.progress = {'G/loss_fake': [],
                         'G/loss_cf': [],
                         'G/loss_total': []}

        if self.encoder_type == 'gcn':
            self.graph_model = DenseGCNConv(self.encode_x_dim, self.encode_h_dim)
        elif self.encoder_type == 'graphConv':
            self.graph_model = DenseGraphConv(self.encode_x_dim, self.encode_h_dim)

        layers = []
        for c0, c1 in zip([self.z_dim + self.encode_h_dim + 1] + self.conv_dims[:-1], self.conv_dims):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(p=self.dropout, inplace=True))
        self.layers = nn.Sequential(*layers)

        self.edges_layer = nn.Linear(self.conv_dims[-1], self.max_num_nodes * self.max_num_nodes)
        self.sigmoid = nn.Sigmoid()

        # 创建损失函数
        self.loss_function = nn.MSELoss()

        # 创建优化器
        self.optimizer = creat_optimizer(self.parameters(), opt=args.opt, lr=self.g_lr)

    def forward(self, adj, features, z, cf_label):
        graph_rep = self.graph_model(features, adj)  # n x num_node x h_dim
        graph_rep = graph_pooling(graph_rep, self.graph_pool_type)
        in_put = torch.cat([z, graph_rep, cf_label], dim=1)
        out_put = self.layers(in_put)
        adj_logits = self.edges_layer(out_put).view(-1, self.max_num_nodes, self.max_num_nodes)
        adj_logits = adj_logits + torch.transpose(adj_logits, dim0=1, dim1=2)
        reconstructed_adj = self.sigmoid(adj_logits)
        a = torch.bernoulli(reconstructed_adj)
        # Generator.postprocess(edges_logits, method=self.post_method)
        return a

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


# 打印验证结果
def print_eval_results(eval_results_all, run_time):
    print(eval_results_all)
    print(r'using time: {}'.format(run_time))
    pass


if __name__ == '__main__':
    args = get_args_for_gcf_gan()
    data_path = get_data_path(args)
    data = load_data(data_path)
    solver = GCFGAN(args, data)
    solver.build_model()
    solver.train()

    print('1')
