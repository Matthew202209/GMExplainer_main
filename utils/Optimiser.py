import torch


def creat_optimizer(parameters, opt='SGD', lr=0.01):
    if opt == 'SGD':
        return torch.optim.SGD(parameters, lr=lr)
    elif opt == 'Adam':
        return torch.optim.Adam(parameters, lr=lr)
