import argparse

def get_args():
    parser = argparse.ArgumentParser()
    #cuda setting
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
