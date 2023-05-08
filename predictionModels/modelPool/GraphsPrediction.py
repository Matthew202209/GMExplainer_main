import random
from numbers import Number
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from torch_geometric.nn import DenseGCNConv, DenseGraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

device = torch.device("cpu")


class GCNN(torch.nn.Module):
    def __init__(
            self,
            num_input,
            num_hidden,
            num_output,
            dropout=0
    ):
        super(GCNN, self).__init__()

        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output

        self.conv1 = DenseGraphConv(num_input, num_hidden)
        self.conv2 = DenseGraphConv(num_hidden, num_hidden)
        self.conv3 = DenseGraphConv(num_hidden, num_hidden)

        self.lin1 = torch.nn.Linear(num_hidden*2, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, num_output)

        self.p = dropout

    def forward(self, x, edge_index, batch=None):

        if batch is None:
            batch = torch.zeros(x.shape[0]).long()

        x = F.relu(self.conv1(x, edge_index))
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        node_embs = x

        x = x1 + x2 + x3

        graph_emb = x

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.p, training=self.training)
        x = F.relu(self.lin2(x))

        x = self.lin3(x)

        return x, (node_embs.detach(), graph_emb.detach())


class Graph_pred_model(nn.Module):
    def __init__(self, x_dim, h_dim, n_out, max_num_nodes, dataset='synthetic'):
        super(Graph_pred_model, self).__init__()
        self.num_graph_models = 3
        self.dataset = dataset
        self.graph_model = nn.ModuleList([DenseGraphConv(x_dim, h_dim) for i in range(self.num_graph_models)])
        self.graph_pool_type = 'mean'
        self.encoder = nn.Sequential(nn.Linear(2 * h_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU())
        self.predictor = nn.Sequential(nn.Linear(h_dim, n_out))
        self.max_num_nodes = max_num_nodes
        self.mask = torch.nn.Parameter(torch.ones(max_num_nodes), requires_grad=True)
        self.register_parameter("mask", self.mask)

    def graph_pooling(self, x, type='mean', mask=None):
        if mask is not None:
            mask_feat = mask.unsqueeze(-1).repeat(1, 1, x.shape[-1])  # batchsize x max_num_node x dim_z
            x = x * mask_feat
        if type == 'max':
            out, _ = torch.max(x, dim=1, keepdim=False)  # dim: the dimension of num_node
        elif type == 'sum':
            out = torch.sum(x, dim=1, keepdim=False)
        elif type == 'mean':
            out = torch.sum(x, dim=1, keepdim=False)
        return out

    def forward(self, x, adj):
        if self.dataset == 'synthetic' or self.dataset == 'community' or self.dataset == 'imdb_b':
            x = torch.ones_like(x).to(device)
        elif self.dataset == 'ogbg_molhiv':
            x = x.clone()
            x[:, :, 2:] = 0.
            x[:, :, 0] = 0.

        mask = torch.zeros(self.max_num_nodes).to(device)
        mask = self.mask.unsqueeze(0).repeat(len(x), 1)  # max_num_nodes -> batch_size x max_num_nodes
        mask = None

        rep_graphs = []
        for i in range(self.num_graph_models):
            rep = self.graph_model[i](x, adj, mask=mask)  # n x num_node x h_dim
            graph_rep = torch.cat(
                [self.graph_pooling(rep, 'mean', mask=mask), self.graph_pooling(rep, 'max', mask=mask)], dim=-1)
            graph_rep = self.encoder(graph_rep)  # n x h_dim
            rep_graphs.append(graph_rep.unsqueeze(0))  # [1 x n x h_dim]

        rep_graph_agg = torch.cat(rep_graphs, dim=0)
        rep_graph_agg = torch.mean(rep_graph_agg, dim=0)  # n x h_dim

        y_pred = self.predictor(rep_graph_agg)  # n x num_class

        return {'y_pred': y_pred, 'rep_graph': rep_graph_agg}
