import torch

from predictionModels.modelPool.GraphsPrediction import GraphPredModel, MolecularClassifier


def build_prediction_model(args, x_dim):
    global pred_model
    if args.prediction_model == 'graph_classification':
        pred_model = GraphPredModel(x_dim, args.p_h_dim, args.p_num_class, args.device, args.dataset_name)
        pred_model.load_state_dict(
            torch.load(args.pred_model_path + f'/prediction/weights_graphPred__{args.dataset_name}' + '.pt'))
    elif args.prediction_model == 'molecule_classification':
        pred_model = MolecularClassifier(x_dim, args.p_h_dim, args.p_num_class)
        pred_model.load_state_dict(
            torch.load(args.pred_model_path + f'/prediction/weights_graphPred__{args.dataset_name}' + '.pt'))
    return pred_model

