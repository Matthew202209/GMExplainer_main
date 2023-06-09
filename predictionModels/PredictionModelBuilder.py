import torch

from predictionModels.modelPool.GraphsPrediction import Graph_pred_model


def build_prediction_model(args, x_dim, max_num_nodes):
    global pred_model
    if args.prediction_model == 'graph_classification':
        pred_model = Graph_pred_model(x_dim, args.p_h_dim, args.p_num_class, max_num_nodes, dataset=args.dataset_name)
        a = torch.load(args.pred_model_path + f'/prediction/weights_graphPred__{args.dataset_name}' + '.pt')
        pred_model.load_state_dict(torch.load(args.pred_model_path + f'/prediction/weights_graphPred__{args.dataset_name}' + '.pt'))
    return pred_model

