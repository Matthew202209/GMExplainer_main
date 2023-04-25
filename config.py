import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # cuda setting
    parser.add_argument('--cuda', type=bool, default=False)
    # datafile setting
    parser.add_argument('--data_path', type=str, default=r'D:\ProjectCodes\GMExplainer\data')
    parser.add_argument('--task_type', type=str, default=r'graph_classification')
    parser.add_argument('--data_type', type=str, default=r'causal')
    parser.add_argument('--dataset_name', type=str, default=r'ogng_molhiv')
    # model checkpoint
    parser.add_argument('--models_save_path', type=str, default=r'.\models_save')
    parser.add_argument('--exp_type', type=str, default=r'prediction')
    # prediction model parameter
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--dim_h', type=int, default=32)
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=int, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='weight decay')
    args = parser.parse_args()
    return args


def get_args_for_gcf_gan():
    parser = argparse.ArgumentParser()
    # device setting
    parser.add_argument('--device', type=str, default="cpu")
    # data loading
    parser.add_argument('--data_path', type=str, default=r'D:\ProjectCodes\GMExplainer\data')
    parser.add_argument('--task_type', type=str, default=r'graph_classification')
    parser.add_argument('--data_type', type=str, default=r'causal')
    parser.add_argument('--dataset_name', type=str, default=r'imdb_m')
    parser.add_argument('--used_dataset', type=str, default=r'CausalDataset')
    # training setting
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--encode_h_dim', type=int, default=32)
    parser.add_argument('--num_epoches_lr_decay', type=int, default=5000)
    parser.add_argument('--lr_update_step', type=int, default=100)
    parser.add_argument('--encode_type', type=str, default= 'graphConv')
    parser.add_argument('--opt', type=str, default='Adam')

    # discriminator setting
    parser.add_argument('--epoches', type=int, default=10000)
    parser.add_argument('--d_train_t', type=int, default=1)
    parser.add_argument('--d_dropout', type=float, default=0.4)
    parser.add_argument('--d_lr', type=float, default=0.001)
    parser.add_argument('--d_graph_pool_type', type=str, default='mean')
    # generator setting
    parser.add_argument('--z_dim', type=int, default=3)
    parser.add_argument('--g_train_t', type=int, default=1)
    parser.add_argument('--conv_dims', default=[64, 128])
    parser.add_argument('--g_dropout', type=float, default=0.2)
    parser.add_argument('--g_lr', type=float, default=0.01)
    parser.add_argument('--post_method', type=str, default='hard_gumbel')
    parser.add_argument('--g_graph_pool_type', type=str, default='mean')
    # prediction model setting
    parser.add_argument('--p_h_dim', type=int, default=32)
    parser.add_argument('--p_num_class', type=int, default=2)
    parser.add_argument('--prediction_model', type=str, default='graph_classification')
    parser.add_argument('--dataset', type=str, default=r'imdb_m')
    parser.add_argument('--pred_model_path', type=str, default=r'D:\ProjectCodes\GMExplainer\models_save')
    # evaluation
    parser.add_argument('--metrics', default=['validity', 'proximity'])

    # save
    parser.add_argument('--model_save_dir', type=str, default=r'D:\ProjectCodes\GMExplainer\models_save\explanation')
    args = parser.parse_args()
    return args


