import argparse


def init_prediction_args():
    parser = argparse.ArgumentParser()
    # cuda setting
    parser.add_argument('--cuda', type=bool, default=False)
    # seed setting
    parser.add_argument('--seed', type=int, default=112)
    # datafile setting
    parser.add_argument('--data_path', type=str, default=r'D:\ProjectCodes\GMExplainer\data')
    parser.add_argument('--task_type', type=str, default=r'graph_classification')
    parser.add_argument('--data_type', type=str, default=r'realworld')
    parser.add_argument('--dataset_name', type=str, default=r'Tox21_ahr')
    # model checkpoint
    parser.add_argument('--models_save_path', type=str, default=r'.\models_save')
    parser.add_argument('--exp_type', type=str, default=r'prediction')
    parser.add_argument('--model', type=str, default='MolecularClassifier')
    parser.add_argument('--metrics', default=['Accuracy', 'AUC-ROC', 'F1-score'])
    parser.add_argument('--save_model', type=bool, default=True)
    # prediction model parameter
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--h_dim', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=int, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='weight decay')
    args = parser.parse_args()
    return args

def init_args_clear():
    parser = argparse.ArgumentParser(description='Graph counterfactual explanation generation')
    # set device
    parser.add_argument('--device', type=str, default="cuda:0")
    # data loading
    parser.add_argument('--data_path', type=str, default=r'D:\ProjectCodes\GMExplainer\data')
    parser.add_argument('--task_type', type=str, default=r'graph_classification')
    parser.add_argument('--data_type', type=str, default=r'causal')
    parser.add_argument('--dataset_name', type=str, default=r'imdb_m')
    parser.add_argument('--used_dataset', type=str, default=r'CausalDataset')

    # prediction_model
    parser.add_argument('--p_h_dim', type=int, default=32)
    parser.add_argument('--p_num_class', type=int, default=2)
    parser.add_argument('--prediction_model', type=str, default='graph_classification')
    parser.add_argument('--pred_model_path', type=str, default=r'D:\ProjectCodes\GMExplainer\models_save')
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=126, metavar='N',
                        help='input batch size for training (default: 500)')  # community: 500， ogbg: 5000
    parser.add_argument('--num_workers', type=int, default=0, metavar='N')
    parser.add_argument('--epochs', type=int, default=5000, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--trn_rate', type=float, default=0.6, help='training data ratio')
    parser.add_argument('--tst_rate', type=float, default=0.2, help='test data ratio')

    parser.add_argument('--disable_u', type=bool, default=True, help='test data ratio')
    parser.add_argument('--dim_z', type=int, default=16)
    parser.add_argument('--dim_h', type=int, default=16, metavar='N', help='dimension of h')

    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=5)

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate for optimizer')  # community: 1e-3
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='weight decay')
    parser.add_argument('--metrics', default=['validity', 'proximity'])

    # expr
    # save
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--model_save_dir', type=str, default=r'D:\ProjectCodes\GMExplainer\models_save\explanation')
    parser.add_argument('--log_save_dir', type=str, default=r'D:\ProjectCodes\GMExplainer\logs')
    parser.add_argument('--results_save_dir', type=str, default=r'D:\ProjectCodes\GMExplainer\results')
    parser.add_argument('--expr', type=int, default=0)

    args = parser.parse_args()
    return args


def init_args_gcf_gan():
    parser = argparse.ArgumentParser()
    # device setting
    parser.add_argument('--device', type=str, default="cuda:0")
    # data loading
    parser.add_argument('--data_path', type=str, default=r'D:\ProjectCodes\GMExplainer\data')
    parser.add_argument('--task_type', type=str, default=r'graph_classification')
    parser.add_argument('--data_type', type=str, default=r'causal')
    parser.add_argument('--dataset_name', type=str, default=r'ogng_molhiv')
    parser.add_argument('--used_dataset', type=str, default=r'CausalDataset')
    # training setting
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--encode_h_dim', type=int, default=32)
    parser.add_argument('--num_epoches_lr_decay', type=int, default=500)
    parser.add_argument('--lr_update_step', type=int, default=100)
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--encode_type', type=str, default='graphConv')
    parser.add_argument('--opt', type=str, default='RMSprop')

    # discriminator setting
    parser.add_argument('--epoches', type=int, default=10000)
    parser.add_argument('--d_train_t', type=int, default=1)
    parser.add_argument('--d_dropout', type=float, default=0.4)
    parser.add_argument('--d_lr', type=float, default=0.001)
    parser.add_argument('--clip_value', type=float, default=0.5)

    parser.add_argument('--d_graph_pool_type', type=str, default='mean')
    # generator setting
    parser.add_argument('--pretrain_epoch', type=int, default=3)
    parser.add_argument('--train_dis_epoch', type=int, default=4)
    parser.add_argument('--z_dim', type=int, default=8)
    parser.add_argument('--n_critic', type=int, default=3)
    parser.add_argument('--conv_dims', default=[64, 128])
    parser.add_argument('--g_dropout', type=float, default=0.2)
    parser.add_argument('--g_lr', type=float, default=0.001)
    parser.add_argument('--lamda_cf', type=float, default=10)
    parser.add_argument('--g_graph_pool_type', type=str, default='mean')
    # prediction model setting
    parser.add_argument('--p_h_dim', type=int, default=32)
    parser.add_argument('--p_num_class', type=int, default=2)
    parser.add_argument('--prediction_model', type=str, default='graph_classification')
    parser.add_argument('--pred_model_path', type=str, default=r'D:\ProjectCodes\GMExplainer\models_save')
    # evaluation
    parser.add_argument('--val_epoch', type=int, default=100)
    parser.add_argument('--metrics', default=['validity', 'proximity'])
    # save
    parser.add_argument('--model_save_dir', type=str, default=r'D:\ProjectCodes\GMExplainer\models_save\explanation')
    parser.add_argument('--log_save_dir', type=str, default=r'D:\ProjectCodes\GMExplainer\logs')
    parser.add_argument('--results_save_dir', type=str, default=r'D:\ProjectCodes\GMExplainer\results')
    parser.add_argument('--expr', type=int, default=0)
    args = parser.parse_args()
    return args


gcfgan_config_dict = {
    'imdb_m': {
        'task_type': r"graph_classification",
        'data_type': r'causal',
        'dataset_name': r'imdb_m',
        'used_dataset': r'CausalDataset',
        # training setting
        'batch_size': 256,
        'num_class': 2,
        'epoches': 15000,
        # discriminator setting
        'd_lr': 0.001,
        # generator setting
        'pretrain_epoch': 4000,
        'train_dis_epoch': 8000,
        'lamda_cf': 1,
        'g_lr': 0.001,
        # prediction model setting
        'p_h_dim': 32,
        'p_num_class': 2,
        'prediction_model': 'graph_classification',
        # save
        'expr': 1},

    'ogng_molhiv': {
        'task_type': r"graph_classification",
        'data_type': r'causal',
        'dataset_name': r'ogng_molhiv',
        'used_dataset': r'CausalDataset',
        # training setting
        'batch_size': 512,
        'num_class': 2,
        'epoches': 10000,
        # discriminator setting
        'd_lr': 0.001,
        # generator setting
        'pretrain_epoch': 4000,
        'train_dis_epoch': 6000,
        'g_lr': 0.001,
        'lamda_cf': 15,
        # prediction model setting
        'p_h_dim': 32,
        'p_num_class': 2,
        'prediction_model': 'graph_classification',
        # save
        'expr': 1},

    'Tox21_ahr': {
        'task_type' :r"graph_classification",
        'data_type' : r'realworld',
       'dataset_name' : r'Tox21_ahr',
        'used_dataset' :r'MolecularDataset',
        # training setting
        'batch_size': 64,
        'num_class': 2,
        'epoches': 5000,
        # discriminator setting
        'd_lr': 0.001,
        # generator setting
        'pretrain_epoch': 2000,
        'train_dis_epoch': 4000,
        'g_lr': 0.001,
        'lamda_cf': 12,
        # prediction model setting
        'p_h_dim': 32,
        'p_num_class': 2,
        'prediction_model': 'molecule_classification',
        # save
        'expr': 1}

}


class PredictionModelConfig:
    Tox21_classifier_config = {
        # cuda setting
        'cuda': True,
        # datafile setting
        'data_path': r'D:\ProjectCodes\GMExplainer\data',
        'task_type': r'graph_classification',
        'data_type': r'realworld',
        'dataset_name': r'Tox21_ahr',
        'dataset_list': [],
        # model checkpoint
        'model' : r'MolecularClassifier',
        'models_save_path': r'.\models_save',
        'exp_type': r'prediction',
        'save_model': True,
        # prediction model parameter
        'batch_size': 64,
        'dim_h': 32,
        'num_class': 2,
        'epochs': 2000,
        'lr': 0.0001,
        'weight_decay': 1e-5,
        'dropout': 0.5}

    Imdb_classifier_config = {
        # cuda setting
        'cuda': True,
        # datafile setting
        'data_path': r'D:\ProjectCodes\GMExplainer\data',
        'task_type': r'graph_classification',
        'data_type': r'causal',
        'dataset_name': r'imdb_m',
        'dataset_list': [],
        # model checkpoint
        'model': r'GraphPredModel',
        'models_save_path': r'.\models_save',
        'exp_type': r'prediction',
        'save_model': True,
        # prediction model parameter
        'batch_size': 64,
        'dim_h': 32,
        'num_class': 2,
        'epochs': 1000,
        'lr': 0.0005,
        'weight_decay': 1e-5,
        'dropout': 0.2}

    Ogng_classifier_config = {
        # cuda setting
        'cuda': True,
        # datafile setting
        'data_path': r'D:\ProjectCodes\GMExplainer\data',
        'task_type': r'graph_classification',
        'data_type': r'causal',
        'dataset_name': r'ogng_molhiv',
        'dataset_list': [],
        # model checkpoint
        'model': r'GraphPredModel',
        'models_save_path': r'.\models_save',
        'exp_type': r'prediction',
        'save_model': True,
        # prediction model parameter
        'batch_size': 512,
        'dim_h': 32,
        'num_class': 2,
        'epochs': 500,
        'lr': 0.0005,
        'weight_decay': 1e-5,
        'dropout': 0.2}
