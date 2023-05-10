import pickle

from config import init_args_gcf_gan, gcfgan_config_dict
from explainers.novel.GANGenerativeCounterfactual import GCFGAN
from utils.LoadData import get_data_path, load_data


class Explainer:
    pass


def read_experiment(args):
    expr_dir = 'D:\ProjectCodes\GMExplainer\data\experiment'
    expr_file = r'{}/expr_{}.pkl'.format(expr_dir,args.dataset_name)
    f = open(expr_file, 'rb')
    expr = pickle.load(f)
    f.close()
    return expr


def get_experiment(args, expr_index):
    exprs = read_experiment(args)
    idx_train_list = exprs['idx_train_list']
    idx_val_list = exprs['idx_val_list']
    idx_test_list = exprs['idx_test_list']
    index_dict = {'train': idx_train_list[expr_index],
                  'val': idx_val_list[expr_index],
                  'test': idx_test_list[expr_index]}
    return index_dict


def get_gcfgan_config(args, config):
    args.dataset_name = config['dataset_name']
    args.task_type = config['task_type']
    args.data_type = config['data_type']
    args.used_dataset = config['used_dataset']
    args.batch_size = config['batch_size']
    args.num_class = config['num_class']
    args.epoches = config['epoches']
    args.d_lr = config['d_lr']
    args.pretrain_epoch = config['pretrain_epoch']
    args.train_dis_epoch = config['train_dis_epoch']
    args.g_lr = config['g_lr']
    args.lamda_cf = config['lamda_cf']
    args.p_h_dim = config['p_h_dim']
    args.p_num_class = config['p_num_class']
    args.prediction_model = config['prediction_model']
    args.expr = config['expr']
    return args


def get_expr_config(args, config, explainer):
    if explainer == 'gcfgan':
        args = get_gcfgan_config(args, config)
    return args


def run_train_explainer(explainer, dataset_name):
    if explainer == 'gcfgan':
        args = get_expr_config(init_args_gcf_gan(), gcfgan_config_dict[dataset_name], explainer)
        index_dict = get_experiment(args, args.expr)
        data_path = get_data_path(args)
        data = load_data(data_path)
        explainer = GCFGAN(args, data)
        explainer.run(index_dict)


if __name__ == '__main__':
    explainers = ['gcfgan']
    run_train_explainer(explainer='gcfgan', dataset_name='Tox21_ahr')




    # 读入实验


    # 选择解释模型
    # 训练需要训练的模型
    # 读入模型进行实验




