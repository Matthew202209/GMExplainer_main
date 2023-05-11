import os
import random
import time
from numbers import Number

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from torch_geometric.nn import DenseGCNConv, DenseGraphConv
from tqdm import tqdm

from Evaluation import evaluate
from dataset.CausalDataset import CausalDataset
from dataset.MolecularDataset import MolecularDataset
from predictionModels.PredictionModelBuilder import build_prediction_model
from utils.LoadData import select_molecular_dataloader, select_dataloader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CLEAR(object):
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
        self.pred_cf_y = None
        # 训练参数
        # 测试参数
        # 模型
        self.clear_explainer = None
        self.pred_model = None
        # 优化器
        self.optimizer = None

        self.test_result = {'validity': [], 'proximity': []}

    def get_prediction(self):
        self.pred_model.eval()
        if self.pred_model is None:
            print(r'No prediction model!')
        else:
            adj = torch.FloatTensor(self.all_data['adj_list']).to(self.device)
            x = torch.FloatTensor(self.all_data['features_list']).to(self.device)
            with torch.no_grad():
                self.pred_y = self.pred_model(x, adj)['y_pred'].argmax(dim=1).view(-1, 1).detach()

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        out = torch.zeros([labels.shape[0]] + [dim]).to(self.device)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        out.scatter_(len(out.size()) - 1, labels, 1.)
        return out

    def setup_seed(self):
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.backends.cudnn.deterministic = True

    def create_data_loader(self, index_dict):
        train_idx, val_idx, test_idx = index_dict['train'], index_dict['val'], index_dict['test']
        val_batch_size = len(val_idx)
        test_batch_size = len(test_idx)

        if self.args.dataset_name == 'Tox21_ahr':
            self.dataloader['train'] = select_molecular_dataloader(self.all_data, train_idx, MolecularDataset,
                                                                   batch_size=self.args.batch_size,
                                                                   num_workers=0)
            self.dataloader['val'] = select_molecular_dataloader(self.all_data, val_idx, MolecularDataset,
                                                                 batch_size=val_batch_size,
                                                                 num_workers=0)
            self.dataloader['test'] = select_molecular_dataloader(self.all_data, test_idx, MolecularDataset,
                                                                  batch_size=test_batch_size,
                                                                  num_workers=0)
        elif self.args.dataset_name == 'ogng_molhiv' or self.args.dataset_name == 'imdb_m':
            self.dataloader['train'] = select_dataloader(self.all_data, train_idx, CausalDataset,
                                                         batch_size=self.args.batch_size,
                                                         num_workers=0)
            self.dataloader['val'] = select_dataloader(self.all_data, val_idx, CausalDataset,
                                                       batch_size=val_batch_size,
                                                       num_workers=0)
            self.dataloader['test'] = select_dataloader(self.all_data, test_idx, CausalDataset,
                                                        batch_size=test_batch_size,
                                                        num_workers=0)

    def build_model(self):
        x_dim = self.all_data["features_list"][0].shape[1]
        max_num_nodes = self.all_data["adj_list"][0].shape[1]
        # 创建判别器和构造器
        self.setup_seed()
        init_params={
            'x_dim': x_dim,
            'max_num_nodes': max_num_nodes
        }
        # 打印模型结构
        self.clear_explainer = GraphCFE(init_params, self.args)

        # 设置训练设备
        self.clear_explainer.to(self.device)

        self.pred_model = build_prediction_model(self.args, x_dim)
        self.pred_model = self.pred_model.to(self.device)
        self.set_optimiser()

    def set_optimiser(self):
        self.optimizer = torch.optim.Adam(self.clear_explainer.parameters(), lr=self.args.lr,
                                          weight_decay=self.args.weight_decay)

    def run(self, index_dict):
        self.create_data_loader(index_dict)
        self.build_model()
        self.get_prediction()
        self.train()
        self.save_test_result()

    def train(self):
        data_loader = self.dataloader['train']
        print('Start training...')
        self.clear_explainer.train()
        self.pred_model.eval()
        # 生成反事实标签
        time_begin = time.time()

        for epoch in tqdm(range(self.args.epochs)):
            loss, loss_kl, loss_sim, loss_cfe, loss_kl_cf = 0.0, 0.0, 0.0, 0.0, 0.0
            batch_num = 0
            for i, element in enumerate(data_loader):
                features = element['features'].float().to(device)
                adj = element['adj'].float().to(device)
                orin_index = element['index']

                p_y_label = self.pred_y[orin_index]
                # 创建标签
                p_y_one_hot = self.label2onehot(p_y_label, self.args.num_class)
                # 创建反事实标签
                p_y_cf_one_hot = CLEAR.create_cf_label(p_y_one_hot).to(torch.float32)
                p_y_cf_label = p_y_cf_one_hot.argmax(dim=1).view(-1, 1)
                self.optimizer.zero_grad()

                # forward pass
                model_return = self.clear_explainer(features, adj, p_y_cf_label)

                # z_cf
                z_mu_cf, z_logvar_cf = self.clear_explainer.get_represent(model_return['features_reconst'],
                                                                          model_return['adj_reconst'], p_y_cf_label)

                # compute loss
                loss_params = {'adj_input': adj, 'features_input': features,
                               'y_cf': p_y_cf_label, 'z_mu_cf': z_mu_cf, 'z_logvar_cf': z_logvar_cf}
                loss_params.update(model_return)

                loss_results = self.compute_loss(loss_params)
                loss_batch, loss_kl_batch, loss_sim_batch, loss_cfe_batch, loss_kl_batch_cf = loss_results['loss'], \
                    loss_results['loss_kl'], loss_results['loss_sim'], loss_results['loss_cfe'], loss_results[
                    'loss_kl_cf']
                loss = loss + loss_batch
                loss_kl = loss_kl + loss_kl_batch
                loss_sim = loss_sim + loss_sim_batch
                loss_cfe = loss_cfe + loss_cfe_batch
                loss_kl_cf = loss_kl_cf + loss_kl_batch_cf
                batch_num = batch_num + 1
                # backward propagation
            loss, loss_kl, loss_sim, loss_cfe, loss_kl_cf = loss / batch_num, loss_kl / batch_num, \
                                                            loss_sim / batch_num, loss_cfe / batch_num, loss_kl_cf / batch_num

            if epoch < 450:
                ((loss_sim + loss_kl + 0 * loss_cfe) / batch_num).backward()
            else:
                ((loss_sim + loss_kl + self.args.alpha * loss_cfe) / batch_num).backward()
            self.optimizer.step()

            # evaluate
            if epoch % 100 == 0:
                eval_results_val = self.evaluation(mode='val')
                eval_results_tst = self.evaluation(mode='test')

                metrics_results_val = ""
                metrics_results_tst = ""
                for k in self.args.metrics:
                    metrics_results_val += f"{k}_val: {eval_results_val[k]:.4f} | "
                    metrics_results_tst += f"{k}_tst: {eval_results_tst[k]:.4f} | "

                print(f"[Train] Epoch {epoch}: train_loss: {(loss):.4f} |" +
                      metrics_results_val + metrics_results_tst +
                      f"time: {(time.time() - time_begin):.4f} |")

                # save
                if self.args.save_model:
                    if epoch % 300 == 0 and epoch > 450:
                        save_dir = r'{}/clear/{}-expr{}'.format(self.args.model_save_dir,
                                                                self.args.dataset_name,
                                                                str(self.args.expr))
                        self.test_result['validity'] = eval_results_tst['validity']
                        self.test_result['proximity'] = eval_results_tst['proximity']
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)

                        CLEAR_path = os.path.join(save_dir, 'best-CLEAR.pt')
                        torch.save(self.clear_explainer.state_dict(), CLEAR_path)
                        print('saved CFE model in: ', CLEAR_path)

    def compute_loss(self, params):
        z_mu, z_logvar, adj_permuted, features_permuted, adj_reconst, features_reconst, \
            adj_input, features_input, y_cf, z_u_mu, z_u_logvar, z_mu_cf, z_logvar_cf = params['z_mu'], params[
            'z_logvar'], params['adj_permuted'], params['features_permuted'], params['adj_reconst'], params[
            'features_reconst'], \
            params['adj_input'], params['features_input'], params['y_cf'], params['z_u_mu'], params['z_u_logvar'], \
            params['z_mu_cf'], params['z_logvar_cf']

        # kl loss
        loss_kl = 0.5 * (((z_u_logvar - z_logvar) + ((z_logvar.exp() + (z_mu - z_u_mu).pow(2)) / z_u_logvar.exp())) - 1)
        loss_kl = torch.mean(loss_kl)

        # similarity loss
        size = len(features_permuted)
        dist_x = torch.mean(distance_feature(features_permuted.view(size, -1), features_reconst.view(size, -1)))
        dist_a = distance_graph_prob(adj_permuted, adj_reconst)

        beta = 10

        loss_sim = beta * dist_x + 10 * dist_a

        # CFE loss #没有使用离散化后的图
        y_pred = self.pred_model(features_reconst, adj_reconst)['y_pred']  # n x num_class
        loss_cfe = F.nll_loss(F.log_softmax(y_pred, dim=-1), y_cf.view(-1).long())

        # rep loss
        if z_mu_cf is None:
            loss_kl_cf = 0.0
        else:
            loss_kl_cf = 0.5 * (((z_logvar_cf - z_logvar) + (
                    (z_logvar.exp() + (z_mu - z_mu_cf).pow(2)) / z_logvar_cf.exp())) - 1)
            loss_kl_cf = torch.mean(loss_kl_cf)

        loss = 1. * loss_sim + 1 * loss_kl + 1.0 * loss_cfe

        loss_results = {'loss': loss, 'loss_kl': loss_kl, 'loss_sim': loss_sim, 'loss_cfe': loss_cfe,
                        'loss_kl_cf': loss_kl_cf}
        return loss_results

    def evaluation(self, mode=f'val'):
        self.clear_explainer.eval()
        data_loader = self.dataloader[mode]

        eval_results_all = {k: 0.0 for k in self.args.metrics}
        size_all = 0
        loss, loss_kl, loss_sim, loss_cfe = 0.0, 0.0, 0.0, 0.0
        batch_num = 0
        for batch_idx, element in enumerate(data_loader):
            features = element['features'].float().to(device)
            adj = element['adj'].float().to(device)
            num_node_real = element['num_node_real'].to(device)
            orin_index = element['index']

            p_y_label = self.pred_y[orin_index]
            # 创建标签
            p_y_one_hot = self.label2onehot(p_y_label, self.args.num_class)
            # 创建反事实标签
            p_y_cf_one_hot = CLEAR.create_cf_label(p_y_one_hot).to(torch.float32)
            p_y_cf_label = p_y_cf_one_hot.argmax(dim=1).view(-1, 1)

            batch_num += 1
            batch_size = element['features'].shape[0]
            size_all += batch_size

            model_return = self.clear_explainer(features, adj, p_y_cf_label)
            adj_reconst, features_reconst = model_return['adj_reconst'], model_return['features_reconst']

            adj_reconst_binary = torch.bernoulli(adj_reconst)

            y_cf_pred = self.pred_model(features_reconst, adj_reconst_binary)['y_pred']
            y_pred = self.pred_model(features, adj)['y_pred']

            # z_cf
            z_mu_cf, z_logvar_cf = None, None

            # compute loss
            loss_params = {'adj_input': adj, 'features_input': features,
                           'y_cf': p_y_cf_label, 'z_mu_cf': z_mu_cf, 'z_logvar_cf': z_logvar_cf}
            loss_params.update(model_return)

            loss_results = self.compute_loss(loss_params)
            loss_batch, loss_kl_batch, loss_sim_batch, loss_cfe_batch = loss_results['loss'], loss_results['loss_kl'], \
                loss_results['loss_sim'], loss_results['loss_cfe']
            loss += loss_batch
            loss_kl += loss_kl_batch
            loss_sim += loss_sim_batch
            loss_cfe += loss_cfe_batch

            # evaluate metrics

            eval_params = {'adj_input': adj, 'x_input': features, 'adj_reconst': adj_reconst_binary,
                           'y_cf': p_y_cf_label, 'y_cf_pred': y_cf_pred, 'num_node_real': num_node_real,
                           'metrics': self.args.metrics, 'device': self.args.device}

            eval_results = evaluate(eval_params)
            for k in self.args.metrics:
                eval_results_all[k] += (batch_size * eval_results[k])

        for k in self.args.metrics:
            eval_results_all[k] /= size_all

        loss, loss_kl, loss_sim, loss_cfe = loss / batch_num, loss_kl / batch_num, loss_sim / batch_num, loss_cfe / batch_num
        eval_results_all['loss'], eval_results_all['loss_kl'], eval_results_all['loss_sim'], eval_results_all[
            'loss_cfe'] = loss, loss_kl, loss_sim, loss_cfe

        return eval_results_all

    def save_test_result(self):
        results_dir = r'{}/clear'.format(self.args.results_save_dir)

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        results_file = r'{}/{}-expr{}.csv'.format(results_dir,
                                                  self.args.dataset_name,
                                                  str(self.args.expr))
        result_df = pd.DataFrame(pd.Series(self.test_result))
        result_df.to_csv(results_file)

    @staticmethod
    def create_cf_label(p_y_label):
        if int(p_y_label.shape[1]) == 2:
            return 1 - p_y_label


class GraphCFE(nn.Module):
    def __init__(self, init_params, args):
        super(GraphCFE, self).__init__()
        self.x_dim = init_params['x_dim']
        self.h_dim = args.dim_h
        self.z_dim = args.dim_z
        self.u_dim = 1  # init_params['u_dim']
        self.dropout = args.dropout
        self.max_num_nodes = init_params['max_num_nodes']
        self.encoder_type = 'gcn'
        self.graph_pool_type = 'mean'
        self.disable_u = args.disable_u
        # self.batch_size = args.batch_size

        if self.disable_u:
            self.u_dim = 0
            print('disable u!')
        if self.encoder_type == 'gcn':
            self.graph_model = DenseGCNConv(self.x_dim, self.h_dim)
        elif self.encoder_type == 'graphConv':
            self.graph_model = DenseGraphConv(self.x_dim, self.h_dim)

        # prior
        self.prior_mean = MLP(self.u_dim, self.z_dim, self.h_dim, n_layers=1, activation='none', slope=.1,
                              device=device)
        self.prior_var = nn.Sequential(
            MLP(self.u_dim, self.z_dim, self.h_dim, n_layers=1, activation='none', slope=.1, device=device),
            nn.Sigmoid())

        # encoder
        self.encoder_mean = nn.Sequential(nn.Linear(self.h_dim + self.u_dim + 1, self.z_dim),
                                          nn.BatchNorm1d(self.z_dim), nn.ReLU())
        self.encoder_var = nn.Sequential(nn.Linear(self.h_dim + self.u_dim + 1, self.z_dim), nn.BatchNorm1d(self.z_dim),
                                         nn.ReLU(), nn.Sigmoid())

        # decoder
        self.decoder_x = nn.Sequential(nn.Linear(self.z_dim + 1, self.h_dim), nn.BatchNorm1d(self.h_dim),
                                       nn.Dropout(self.dropout), nn.ReLU(),
                                       nn.Linear(self.h_dim, self.h_dim), nn.BatchNorm1d(self.h_dim),
                                       nn.Dropout(self.dropout), nn.ReLU(),
                                       nn.Linear(self.h_dim, self.max_num_nodes * self.x_dim))
        self.decoder_a = nn.Sequential(nn.Linear(self.z_dim + 1, self.h_dim), nn.BatchNorm1d(self.h_dim),
                                       nn.Dropout(self.dropout), nn.ReLU(),
                                       nn.Linear(self.h_dim, self.h_dim), nn.BatchNorm1d(self.h_dim),
                                       nn.Dropout(self.dropout), nn.ReLU(),
                                       nn.Linear(self.h_dim, self.max_num_nodes * self.max_num_nodes), nn.Sigmoid())
        self.graph_norm = nn.BatchNorm1d(self.h_dim)

    def encoder(self, features, adj, y_cf, u=None):
        # Q(Z|X,U,A,Y^CF)
        # input: x, u, A, y^cf
        # output: z
        features = features.to(torch.float32)
        adj = adj.to(torch.float32)



        graph_rep = self.graph_model(features, adj)  # n x num_node x h_dim

        graph_rep = self.graph_pooling(graph_rep, self.graph_pool_type)  # n x h_dim
        # graph_rep = self.graph_norm(graph_rep)

        if u is None:
            z_mu = self.encoder_mean(torch.cat((graph_rep, y_cf), dim=1))
            z_logvar = self.encoder_var(torch.cat((graph_rep, y_cf), dim=1))
        else:
            z_mu = self.encoder_mean(torch.cat((graph_rep, u, y_cf), dim=1))
            z_logvar = self.encoder_var(torch.cat((graph_rep, u, y_cf), dim=1))

        return z_mu, z_logvar

    def get_represent(self, features, adj, y_cf):
        # encoder
        z_mu, z_logvar = self.encoder(features,  adj, y_cf)

        return z_mu, z_logvar

    def decoder(self, z, y_cf):
        if self.disable_u:
            adj_reconst = self.decoder_a(torch.cat((z, y_cf), dim=1)).view(-1, self.max_num_nodes,
                                                                           self.max_num_nodes)
        else:
            adj_reconst = self.decoder_a(torch.cat((z, y_cf), dim=1)).view(-1, self.max_num_nodes, self.max_num_nodes)

        features_reconst = self.decoder_x(torch.cat((z, y_cf), dim=1)).view(-1, self.max_num_nodes, self.x_dim)
        return features_reconst, adj_reconst

    def graph_pooling(self, x, type='mean'):
        if type == 'max':
            out, _ = torch.max(x, dim=1, keepdim=False)
        elif type == 'sum':
            out = torch.sum(x, dim=1, keepdim=False)
        elif type == 'mean':
            out = torch.sum(x, dim=1, keepdim=False)
        return out

    def prior_params(self, u, batch_size):  # P(Z|U)
        if u is None:
            z_u_mu = torch.zeros((batch_size, self.h_dim)).to(device)
            z_u_logvar = torch.ones((batch_size, self.h_dim)).to(device)
        else:
            z_u_logvar = self.prior_var(u)
            z_u_mu = self.prior_mean(u)
        return z_u_mu, z_u_logvar

    def reparameterize(self, mu, logvar):
        '''
        compute z = mu + std * epsilon
        '''
        if self.training:
            # compute the standard deviation from logvar
            std = torch.exp(0.5 * logvar)
            # sample epsilon from a normal distribution with mean 0 and
            # variance 1
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def score(self):
        return

    def forward(self, features, adj, y_cf, u=None):
        batch_size = features.shape[0]
        if u is not None:
            u_onehot = u
            z_u_mu, z_u_logvar = self.prior_params(u_onehot, batch_size)
            z_mu, z_logvar = self.encoder(features, adj, y_cf, u=u_onehot)
            z_sample = self.reparameterize(z_mu, z_logvar)
            # decoder
            features_reconst, adj_reconst = self.decoder(z_sample, y_cf)
            return {'z_mu': z_mu, 'z_logvar': z_logvar, 'adj_permuted': adj, 'features_permuted': features,
                    'adj_reconst': adj_reconst, 'features_reconst': features_reconst, 'z_u_mu': z_u_mu,
                    'z_u_logvar': z_u_logvar}
        else:
            z_u_mu, z_u_logvar = self.prior_params(u, batch_size)
            z_mu, z_logvar = self.encoder(features, adj, y_cf)
            z_sample = self.reparameterize(z_mu, z_logvar)
            # decoder
            features_reconst, adj_reconst = self.decoder(z_sample, y_cf)
            return {'z_mu': z_mu, 'z_logvar': z_logvar, 'adj_permuted': adj, 'features_permuted': features,
                    'adj_reconst': adj_reconst, 'features_reconst': features_reconst, 'z_u_mu': z_u_mu,
                    'z_u_logvar': z_u_logvar}


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation='none', slope=.1, device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        if isinstance(hidden_dim, Number):
            self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError('Wrong argument type for hidden_dim: {}'.format(hidden_dim))

        if isinstance(activation, str):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.hidden_dim = activation
        else:
            raise ValueError('Wrong argument type for activation: {}'.format(activation))

        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
        self.fc = nn.ModuleList(_fc_list)
        self.to(self.device)

    @staticmethod
    def xtanh(x, alpha=.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f[c](self.fc[c](h))
        return h


def distance_feature(feat_1, feat_2):
    pdist = nn.PairwiseDistance(p=2)
    output = pdist(feat_1, feat_2) / 4
    return output


def distance_graph_prob(adj_1, adj_2_prob):
    dist = F.binary_cross_entropy(adj_2_prob, adj_1)
    return dist
