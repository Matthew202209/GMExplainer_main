import argparse
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score

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
from utils.LoadData import get_data_path, load_data, select_dataloader, create_experiment
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
        self.dataloader = {}

        # 模型预测结果
        self.pred_y = None
        # 训练参数
        # 测试参数
        # 模型
        self.D = None
        self.G = None
        self.P = None
        self.pred_model = None
        # 地址
        # self.model_save_dir = self.args.model_save_dir

    def get_prediction(self):
        if self.pred_model is None:
            print(r'No prediction model!')
        else:
            adj = torch.DoubleTensor(self.all_data['adj_list']).to(self.device)
            x = torch.DoubleTensor(self.all_data['features_list']).to(self.device)
            with torch.no_grad():
                self.pred_y = self.pred_model(x, adj)['y_pred'].argmax(dim=1).view(-1, 1).detach()

    def setup_seed(self):
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.backends.cudnn.deterministic = True

    """数据处理"""

    # 载入数据

    # 创建data_loader

    def create_data_loader(self, index_dict):
        _package_ = __import__("dataset.{}".format(self.args.used_dataset), fromlist=[self.args.used_dataset])
        dataset = getattr(_package_, self.args.used_dataset)
        for mode in ['train', 'val', 'test']:
            data_idx = index_dict[mode]
            if mode == 'train':
                self.dataloader['train'] = select_dataloader(self.all_data, data_idx, dataset,
                                                             batch_size=self.args.batch_size,
                                                             num_workers=0, padded=False)
            elif mode == 'val':
                self.dataloader['val'] = select_dataloader(self.all_data, data_idx, dataset, batch_size=len(data_idx),
                                                           num_workers=0, padded=False)
            elif mode == 'test':
                self.dataloader['test'] = select_dataloader(self.all_data, data_idx, dataset, batch_size=len(data_idx),
                                                            num_workers=0, padded=False)

    def build_model(self):
        x_dim = self.all_data["features_list"][0].shape[1]
        max_num_nodes = self.all_data["adj_list"][0].shape[1]
        # 创建判别器和构造器
        self.setup_seed()
        self.D = Discriminator(self.args, x_dim)
        self.setup_seed()
        self.G = Generator(self.args, max_num_nodes, x_dim)

        # 打印模型结构
        GCFGAN.print_network(self.G, 'G')
        GCFGAN.print_network(self.D, 'D')

        # 设置训练设备
        self.D.to(self.device)
        self.G.to(self.device)

        self.pred_model = build_prediction_model(self.args, x_dim, max_num_nodes)
        self.pred_model = self.pred_model.to(self.device)

    def test(self):
        pass

    def train(self):
        global d_loss_real, d_loss_fake, d_loss, g_loss_cfe, g_loss, \
            g_dis_loss, adj, x, adj_reconst, orin_index, g_loss_fake_ture, p_y_cf_label
        data_loader = self.dataloader['train']
        best_validity = -1
        print('Start training...')
        # 生成反事实标签
        time_begin = time.time()
        best_loss = 1000000000000
        for epoch in tqdm(range(self.args.epoches)):
            self.D.train()
            self.G.train()
            self.pred_model.eval()
            # =================================================================================== #
            #                             1. Prepare data                                         #
            # =================================================================================== #

            for i, element in enumerate(data_loader):
                adj, x, orin_index = element['adj'], element['features'], element['index']
                adj = adj.to(self.device)
                x = x.to(self.device)
                orin_index = orin_index.to(self.device)
                p_y_label = self.pred_y[orin_index]
                # 创建标签
                p_y_label = self.label2onehot(p_y_label, self.args.num_class)
                # 创建反事实标签
                p_y_cf_label = GCFGAN.create_cf_label(p_y_label)

                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #
                # 生成假样本

                z = self.sample_z(self.args.batch_size)
                adj_reconst = self.G(adj, x, z, p_y_cf_label)

                # 计算真样本正例
                logits_real_ture = self.D(adj, x, p_y_label)
                d_loss_real_ture = - torch.mean(logits_real_ture)

                # 计算真样本负例
                logits_real_false = self.D(adj, x, p_y_cf_label)
                d_loss_real_false = torch.mean(logits_real_false)

                # 计算创造数据的loss
                logits_fake = self.D(adj_reconst.detach().to(self.device), x, p_y_cf_label)
                d_loss_fake = torch.mean(logits_fake)
                d_loss = d_loss_real_ture + d_loss_real_false + d_loss_fake
                # 更新参数
                self.reset_grad()
                d_loss.backward()
                self.D.optimizer.step()

                for p in self.D.parameters():
                    p.data.clamp_(-self.args.clip_value, self.args.clip_value)
                self.D.progress['D/d_loss_real_ture'] += [d_loss_real_ture.item()]
                self.D.progress['D/d_loss_real_false'] += [d_loss_real_false.item()]
                self.D.progress['D/d_loss_fake'] += [d_loss_fake.item()]
                self.D.progress['D/loss_total'] += [d_loss.item()]

                # =================================================================================== #
                #                             3. Train the generator                                  #
                # =================================================================================== #

                if (i + 1) % self.args.n_critic == 0:
                    if (epoch + 1) < self.args.pretrain_epoch:
                        z = self.sample_z(self.args.batch_size)
                        adj_reconst = self.G(adj, x, z, p_y_cf_label)
                        logits_fake = self.D(adj_reconst, x, p_y_cf_label)
                        g_loss = -torch.mean(logits_fake)
                        if 'G/g_loss' not in self.G.progress.keys():
                            self.G.progress['G/g_loss'] = []
                            self.G.progress['G/g_loss'] += [g_loss.item()]
                        else:
                            self.G.progress['G/g_loss'] += [g_loss.item()]
                    else:
                        if (epoch + 1) == self.args.pretrain_epoch:
                            print('Finished Pretraining')
                            print('Start CF Training......')
                        z = self.sample_z(self.args.batch_size)
                        adj_reconst = self.G(adj, x, z, p_y_cf_label)
                        # 假样本预测结果
                        p_y_pred_fake = self.pred_model(x, adj_reconst)['y_pred']  # n x num_class

                        loss_cf = F.nll_loss(F.log_softmax(p_y_pred_fake, dim=-1),
                                             p_y_cf_label.argmax(dim=1).view(-1, 1).detach().view(-1).long())

                        # p_y_cf_label = self.label2onehot(p_y_cf_label, self.args.num_class)
                        # 计算对抗损失
                        logits_fake_ture = self.D(adj_reconst, x, p_y_cf_label)
                        g_loss_fake_ture = -torch.mean(logits_fake_ture)
                        g_dis_loss = distance_graph_prob(adj, adj_reconst)

                        self.G.progress['G/g_loss'] += [g_loss_fake_ture.item()]
                        if (epoch + 1) < self.args.train_dis_epoch:
                            g_loss = 0 * loss_cf + g_dis_loss + g_loss_fake_ture
                            if 'G/dis_loss' not in self.G.progress.keys():
                                self.G.progress['G/dis_loss'] = []
                                self.G.progress['G/dis_loss'] += [g_dis_loss.item()]
                            else:
                                self.G.progress['G/dis_loss'] += [g_dis_loss.item()]
                        else:
                            g_loss = loss_cf + g_dis_loss + g_loss_fake_ture
                            if 'G/loss_cf' not in self.G.progress.keys():
                                self.G.progress['G/loss_cf'] = []
                                self.G.progress['G/loss_cf'] += [loss_cf.item()]
                                self.G.progress['G/dis_loss'] += [g_dis_loss.item()]
                            else:
                                self.G.progress['G/loss_cf'] += [loss_cf.item()]
                                self.G.progress['G/dis_loss'] += [g_dis_loss.item()]
                    self.reset_grad()
                    g_loss.backward()
                    self.G.optimizer.step()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #
            if (epoch + 1) % self.args.val_epoch == 0:
                et = time.time() - time_begin
                y_cf = p_y_cf_label.argmax(dim=1).view(-1, 1).detach()
                c_adj = torch.bernoulli(adj_reconst)
                with torch.no_grad():
                    y_cf_pred = self.pred_model(x, c_adj)['y_pred']
                eval_params = {'pred_model': self.pred_model, 'adj_input': adj, 'x_input': x, 'adj_reconst': c_adj,
                               'y_cf': y_cf, 'y_cf_pred': y_cf_pred,
                               'metrics': self.args.metrics, 'device': self.args.device}
                train_eval_results = evaluate(eval_params)
                train_eval_results.update({'D/d_loss': self.D.progress['D/loss_total'][-1]})
                train_eval_results.update({'G/g_loss': self.G.progress['G/g_loss'][-1]})

                if (epoch + 1) > self.args.pretrain_epoch:
                    if (epoch + 1) < self.args.train_dis_epoch:
                        train_eval_results.update({'G/dis_loss': self.G.progress['G/dis_loss'][-1]})
                    else:
                        train_eval_results.update({'G/dis_loss': self.G.progress['G/dis_loss'][-1],
                                                   'G/loss_cf': self.G.progress['G/loss_cf'][-1]})

                print(r'epoch:{}'.format(epoch + 1))
                print(r'time:{}'.format(et))
                p = r'train: '
                for k, v in train_eval_results.items():
                    p = p + f"{k}: {v:.4f} | "
                print(p)

                val_eval_results = self.evaluation(mode=f'val')
                _ = self.evaluation(mode=f'test')

                total_loss = val_eval_results['total_loss']
                if (epoch + 1) >= self.args.train_dis_epoch:
                    save_dir = r'{}/{}-expr{}'.format(self.args.model_save_dir,
                                                       self.args.dataset_name,
                                                       self.args.expr)

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    if total_loss < best_loss:
                        best_loss = total_loss
                        G_path = os.path.join(save_dir, 'best-G.pt')
                        D_path = os.path.join(save_dir, 'best-D.pt')
                        torch.save(self.G.state_dict(), G_path)
                        torch.save(self.D.state_dict(), D_path)
                        print('Saved model checkpoints into {}...'.format(self.args.model_save_dir))
                time_begin = time.time()
                # =================================================================================== #
                #                              5. Update Learning Rate                                #
                # =================================================================================== #
            if (epoch + 1) % self.args.lr_update_step == 0 and (epoch + 1) > (
                    self.args.epoches - self.args.num_epoches_lr_decay):
                self.G.g_lr -= (self.args.g_lr / float(self.args.num_epoches_lr_decay))
                self.D.d_lr -= (self.args.d_lr / float(self.args.num_epoches_lr_decay))
                self.update_lr()
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(self.G.g_lr, self.D.d_lr))

    def evaluation(self, mode= f'val'):
        # 调整模型状态
        self.D.eval()
        self.G.eval()

        eval_results_all = {}
        data_loader = self.dataloader[mode]
        with torch.no_grad():
            for i, element in enumerate(data_loader):
                adj, x, orin_index = element['adj'], element['features'], element['index']
                adj = adj.to(self.device)
                x = x.to(self.device)
                # 得到反事实标签
                batch_size = adj.shape[0]
                p_y_label = self.pred_y[orin_index]
                p_y_label = self.label2onehot(p_y_label, self.args.num_class)
                # 创建反事实标签
                y_cf_label = self.create_cf_label(p_y_label)
                y_cf = y_cf_label.argmax(dim=1).view(-1, 1)

                # 生成cf图
                self.setup_seed()
                z = self.sample_z(batch_size).to(self.device)
                adj_reconst = self.G(adj, x, z, y_cf_label)

                # 计算loss

                g_dis_loss = distance_graph_prob(adj, adj_reconst)
                p_y_pred_test = self.pred_model(x, adj_reconst)['y_pred']  # n x num_class
                loss_cf = F.nll_loss(F.log_softmax(p_y_pred_test, dim=-1),
                                     y_cf.detach().view(-1).long())
                c_adj = torch.bernoulli(adj_reconst)
                logits_fake_ture = self.D(c_adj, x, y_cf_label)
                g_loss_fake_ture = -torch.mean(logits_fake_ture)

                total_loss = g_dis_loss + loss_cf + g_loss_fake_ture

                eval_results_all['g_loss_val'] = g_loss_fake_ture.item()
                eval_results_all['loss_cf'] = loss_cf.item()
                eval_results_all['g_dis_loss'] = g_dis_loss.item()
                eval_results_all['total_loss'] = total_loss.item()

                with torch.no_grad():
                    y_cf_pred = self.pred_model(x, c_adj)['y_pred']

                # 计算评价指标
                eval_params = {'pred_model': self.pred_model, 'adj_input': adj, 'x_input': x, 'adj_reconst': c_adj,
                               'y_cf': y_cf, 'y_cf_pred': y_cf_pred,
                               'metrics': self.args.metrics, 'device': self.args.device}

                eval_results = evaluate(eval_params)
                eval_results_all.update(eval_results)

                p = mode + ':  '
                for k in eval_results_all.keys():
                    p =  p + f"{k}: {eval_results_all[k]:.4f} | "
                print(p)

        return eval_results_all

    def run(self, index_dict):
        self.create_data_loader(index_dict)
        self.build_model()
        self.get_prediction()
        self.train()

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

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        out = torch.zeros([labels.shape[0]] + [dim]).to(self.device)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        out.scatter_(len(out.size()) - 1, labels, 1.)
        return out

    @staticmethod
    def print_network(model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    @staticmethod
    def create_cf_label(p_y_label):
        if int(p_y_label.shape[1]) == 2:
            return 1 - p_y_label


class Discriminator(nn.Module):
    def __init__(self, args, x_dim):
        super(Discriminator, self).__init__()

        # 参数设置
        self.encode_x_dim = x_dim
        self.encode_h_dim = args.encode_h_dim
        self.encoder_type = args.encode_type
        self.num_class = args.num_class
        self.graph_pool_type = args.d_graph_pool_type
        self.dropout = args.d_dropout
        self.d_lr = args.d_lr
        # 计数器和进程记录
        self.counter = 0
        self.progress = {'D/d_loss_real_ture': [],
                         'D/d_loss_real_false': [],
                         'D/d_loss_fake': [],
                         'D/loss_total': []
                         }

        if self.encoder_type == 'gcn':
            self.graph_model = DenseGCNConv(self.encode_x_dim, self.encode_h_dim)
        elif self.encoder_type == 'graphConv':
            self.graph_model = DenseGraphConv(self.encode_x_dim, self.encode_h_dim)

        self.label_encoder = nn.Linear(self.num_class, 16)

        self.linear_layer = nn.Sequential(
            nn.Linear(self.encode_h_dim + 16, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=self.dropout, inplace=True),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(p=self.dropout, inplace=True),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )
        # 创建优化器
        self.optimizer = creat_optimizer(self.parameters(), opt=args.opt, lr=self.d_lr)

    def forward(self, adj, features, one_hot_label):
        graph_rep = self.graph_model(features, adj)  # n x num_node x h_dim
        graph_rep = graph_pooling(graph_rep, self.graph_pool_type)
        encode_label = self.label_encoder(one_hot_label)
        in_put = torch.cat([graph_rep, encode_label], dim=-1)

        output = self.linear_layer(in_put)
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
        self.num_class = args.num_class
        self.graph_pool_type = args.g_graph_pool_type
        self.dropout = args.g_dropout
        self.g_lr = args.g_lr

        # 计数器和进程记录
        self.counter = 0
        self.progress = {'G/loss_fake': [],
                         'G/loss_dis': [],
                         'G/loss_cf': [],
                         'G/loss_total': []
                         }

        if self.encoder_type == 'gcn':
            self.graph_model = DenseGCNConv(self.encode_x_dim, self.encode_h_dim)
        elif self.encoder_type == 'graphConv':
            self.graph_model = DenseGraphConv(self.encode_x_dim, self.encode_h_dim)

        self.label_encoder = nn.Linear(self.num_class, 32)

        layers = []
        for c0, c1 in zip([self.z_dim + self.encode_h_dim + 32] + self.conv_dims[:-1], self.conv_dims):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.LayerNorm(c1))
            layers.append(nn.Dropout(p=self.dropout, inplace=True))
        self.layers = nn.Sequential(*layers)

        self.edges_layer = nn.Linear(self.conv_dims[-1], self.max_num_nodes * self.max_num_nodes)
        self.sigmoid = nn.Sigmoid()

        # 创建优化器
        self.optimizer = creat_optimizer(self.parameters(), opt=args.opt, lr=self.g_lr)

    def forward(self, adj, features, z, cf_label):
        # encode label
        encode_label = self.label_encoder(cf_label)
        # graph rep
        rep = self.graph_model(features, adj)  # n x num_node x h_dim
        graph_rep = graph_pooling(rep, self.graph_pool_type)

        in_put = torch.cat([z, graph_rep, encode_label], dim=-1)
        out_put = self.layers(in_put)
        adj_logits = self.edges_layer(out_put).view(-1, self.max_num_nodes, self.max_num_nodes)

        # 对称
        adj_logits = (adj_logits + torch.transpose(adj_logits, dim0=1, dim1=2)) / 2
        reconstructed_adj = self.sigmoid(adj_logits)

        # Generator.postprocess(edges_logits, method=self.post_method)
        return reconstructed_adj

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


def distance_graph_prob(adj_1, adj_2_prob):
    dist = F.binary_cross_entropy(adj_2_prob, adj_1)
    return dist


if __name__ == '__main__':
    num_expr = 10
    args = get_args_for_gcf_gan()
    data_path = get_data_path(args)
    data = load_data(data_path)
    num_data = len(data['adj_list'])
    idx_train_list, idx_val_list, idx_test_list = create_experiment(num_expr, num_data)
    for expr in range(num_expr):
        index_dict = {'train': idx_train_list[expr],
                      'val': idx_val_list[expr],
                      'test': idx_test_list[expr]}
        solver = GCFGAN(args, data)
        solver.run(index_dict)


