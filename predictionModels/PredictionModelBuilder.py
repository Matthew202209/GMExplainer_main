import torch

from predictionModels.modelPool.GraphsPrediction import Graph_pred_model


def build_prediction_model(args, x_dim, max_num_nodes):
    global pred_model
    if args.prediction_model == 'graph_classification':
        pred_model = Graph_pred_model(x_dim, args.p_h_dim, args.p_n_out, max_num_nodes, dataset=args.dataset)
        pred_model.load_state_dict(torch.load(args.pred_model_path + f'prediction/weights_graphPred__{args.dataset}' + '.pt'))
        pred_model.eval()
    return pred_model
