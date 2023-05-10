# Based on https://github.com/RexYing/gnn-model-explainer/blob/master/explainer/explain.py

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from torch.nn.parameter import Parameter


class Perturb(nn.Module):
    def __init__(self, adj, edge_additions):
        super(Perturb, self).__init__()

        self.P = None
        self.adj = adj
        self.num_nodes = self.adj.shape[0]
        self.edge_additions = edge_additions  # are edge additions included in perturbed matrix

        # P_hat needs to be symmetric ==> learn vector representing entries in upper/lower triangular matrix and use to populate P_hat later
        self.P_vec_size = int((self.num_nodes * self.num_nodes - self.num_nodes) / 2) + self.num_nodes

        if self.edge_additions:
            self.P_vec = Parameter(torch.FloatTensor(torch.zeros(self.P_vec_size)))
        else:
            self.P_vec = Parameter(torch.FloatTensor(torch.ones(self.P_vec_size)))
        self.P_hat_symm = None
        self.reset_parameters()

    def reset_parameters(self, eps=10 ** -4):
        # Think more about how to initialize this
        with torch.no_grad():
            if self.edge_additions:
                adj_vec = create_vec_from_symm_matrix(self.adj).numpy()
                for i in range(len(adj_vec)):
                    if i < 1:
                        adj_vec[i] = adj_vec[i] - eps
                    else:
                        adj_vec[i] = adj_vec[i] + eps
                torch.add(self.P_vec, torch.FloatTensor(adj_vec))  # self.P_vec is all 0s
            else:
                torch.sub(self.P_vec, eps)

    def forward(self, adj):
        # Same as normalize_adj in utils.py except includes P_hat in A_tilde
        self.P_hat_symm = create_symm_matrix_from_vec(self.P_vec, self.num_nodes)  # Ensure symmetry

        A_tilde = torch.FloatTensor(self.num_nodes, self.num_nodes)
        A_tilde.requires_grad = True

        if self.edge_additions:  # Learn new adj matrix directly
            A_tilde = F.sigmoid(self.P_hat_symm) + torch.eye(self.num_nodes)  # Use sigmoid to bound P_hat in [0,1]
        else:  # Learn P_hat that gets multiplied element-wise with adj -- only edge deletions
            A_tilde = F.sigmoid(self.P_hat_symm) * adj

        return A_tilde

    def discrete(self):
        self.P = (F.sigmoid(self.P_hat_symm) >= 0.5).float()  # threshold P_hat
        if self.edge_additions:
            A_tilde = self.P + torch.eye(self.num_nodes)
        else:
            A_tilde = self.P * self.adj + torch.eye(self.num_nodes)
        return A_tilde, self.P


class CFExplainer:

    def __init__(self, args, pre_model, adj, features):
        super(CFExplainer, self).__init__()

        self.pre_model = pre_model
        self.pre_model.eval()
        self.args = args
        self.adj = adj
        self.features = features
        self.beta = args.beta
        self.num_classes = args.num_classes
        self.device = args.device
        self.lr = args.lr
        self.edge_additions =args.edge_additions
        self.perturb = Perturb(self.adj, args.edge_additions)
        self.pre_y = None
        self.cf_pre_y = None
        self.cf_optimizer = None
        self.P = None

    def set_cf_optimizer(self):
        if self.args.cf_optimizer_type == "SGD" and self.args.n_momentum == 0.0:
            self.cf_optimizer = optim.SGD(self.perturb.parameters(), lr=self.lr)
        elif self.args.cf_optimizer_type == "SGD" and self.args.n_momentum != 0.0:
            self.cf_optimizer = optim.SGD(self.perturb.parameters(), lr=self.lr,
                                          nesterov=True, momentum=self.args.n_momentum.n_momentum)
        elif self.args.cf_optimizer_type == "Adadelta":
            self.cf_optimizer = optim.Adadelta(self.perturb.parameters(), lr=self.lr)

    def get_origin_y(self):
        self.pre_y = self.pre_model(self.features, self.adj)['y_pred'].argmax(dim=1).view(-1, 1).detach()

    def create_cf_label(self):
        if self.num_classes == 2:
            self.cf_pre_y = 1 - self.num_classes

    def explain(self):
        self.get_origin_y()
        self.create_cf_label()
        best_cf_example = None
        best_loss = np.inf
        for epoch in range(self.args.num_epochs):
            new_example, loss_total = self.train(epoch)
            if new_example != [] and loss_total < best_loss:
                best_cf_example = new_example[0]
                best_loss = loss_total

        return best_cf_example

    def loss(self, output, y_pred_orig, y_pred_new_actual):
        pred_same = (y_pred_new_actual == y_pred_orig).float()

        # Need dim >=2 for F.nll_loss to work
        output = output.unsqueeze(0)
        y_pred_orig = y_pred_orig.unsqueeze(0)

        if self.edge_additions:
            cf_adj = self.P
        else:
            cf_adj = self.P * self.adj
        cf_adj.requires_grad = True  # Need to change this otherwise loss_graph_dist has no gradient

        # Want negative in front to maximize loss instead of minimizing it to find CFs
        loss_pred = - F.nll_loss(output, y_pred_orig)
        loss_graph_dist = sum(sum(abs(cf_adj - self.adj))) / 2  # Number of edges changed (symmetrical)

        # Zero-out loss_pred with pred_same if prediction flips
        loss_total = pred_same * loss_pred + self.beta * loss_graph_dist
        return loss_total, loss_pred, loss_graph_dist, cf_adj

    def train(self, epoch):
        global cf_stats
        t = time.time()
        self.perturb.train()
        self.cf_optimizer.zero_grad()
        p_adj = self.perturb.forward(self.adj)
        p_discrete_adj, self.P = self.perturb.discrete()

        output = self.pre_model(self.features, p_adj)
        output_actual = self.pre_model(self.features, p_discrete_adj)
        y_pred_new = torch.argmax(output)
        y_pred_new_actual = torch.argmax(output_actual)
        loss_total, loss_pred, loss_graph_dist, cf_adj = self.loss(output,  self.pre_y, y_pred_new_actual)

        loss_total.backward()
        clip_grad_norm(self.perturb.parameters(), 2.0)
        self.cf_optimizer.step()
        if y_pred_new_actual != self.pre_y or epoch == self.args.num_epochs:
            cf_stats = [cf_adj.detach().numpy(), self.adj.detach().numpy(),
                        self.pre_y.item(), y_pred_new.item(),
                        y_pred_new_actual.item(),
                        self.adj.shape[0], loss_total.item(),
                        loss_pred.item(), loss_graph_dist.item(), epoch]
        return cf_stats, loss_total.item()


def create_vec_from_symm_matrix(matrix):
    idx = torch.tril_indices(matrix.shape[0], matrix.shape[0])
    vector = matrix[idx[0], idx[1]]
    return vector


def create_symm_matrix_from_vec(vector, n_rows):
    matrix = torch.zeros(n_rows, n_rows)
    idx = torch.tril_indices(n_rows, n_rows)
    matrix[idx[0], idx[1]] = vector
    symm_matrix = torch.tril(matrix) + torch.tril(matrix, -1).t()
    return symm_matrix
