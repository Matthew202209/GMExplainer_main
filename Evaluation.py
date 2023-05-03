import torch
from torch.nn import functional as F


def evaluate(eval_params):
    eval_results = {}
    pred_model, adj_input, adj_reconst, x, y_cf, y_cf_pred, metrics, device = eval_params['pred_model'], eval_params['adj_input'], \
        eval_params['adj_reconst'], eval_params['x_input'], eval_params['y_cf'], eval_params['y_cf_pred'], \
        eval_params['metrics'], eval_params['device']

    if 'validity' in metrics:
        score_valid = evaluate_validity(y_cf, y_cf_pred, device)
        eval_results['validity'] = score_valid.item()
    if 'proximity' in metrics:
        score_proximity = evaluate_proximity(adj_input, adj_reconst)
        eval_results['proximity'] = score_proximity.item()

    return eval_results


def evaluate_validity(y_cf, y_cf_pred, device):
    y_cf_pred_binary = F.softmax(y_cf_pred, dim=-1)
    y_cf_pred_binary = y_cf_pred_binary.argmax(dim=1).view(-1, 1)
    y_cf = y_cf.argmax(dim=1).view(-1, 1)
    y_eq = torch.where(y_cf == y_cf_pred_binary, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))
    score_valid = torch.mean(y_eq)
    return score_valid


def evaluate_proximity(adj_input, adj_reconst):
    score_proximity = (adj_input == adj_reconst).float().mean()
    return score_proximity
