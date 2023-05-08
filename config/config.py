#
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
        'models_save_path': r'.\models_save',
        'exp_type': r'prediction',
        # prediction model parameter
        'batch_size': 256,
        'dim_h': 32,
        'num_class': 2,
        'epochs': 2000,
        'lr': 0.001,
        'weight_decay': 1e-5}
