import pickle
import random

import numpy as np

path = r'D:\ProjectCodes\GMExplainer\GMExplainer\data\graph_classification\causal\imdb_m.pickle'
if __name__ == '__main__':
    with open('D:\ProjectCodes\GMExplainer\GMExplainer\data\graph_classification\causal\imdb_m.pickle',
              'rb') as handle:
        data = pickle.load(handle)['data']
    data_dict = {}
    adj_list = data.adj_all
    features = data.feature_all
    labels = data.labels_all
    index = data.index
    num = len(index)
    ran = random.sample(range(num), num)

    train_index = ran[0:int(num * 0.6)]
    val_index = ran[int(num * 0.6) + 1:int(num * 0.8)]
    test_index = ran[int(num * 0.8) + 1: num - 1]
    node_num = np.array(data.len_all)
    train_node_num = node_num[train_index].tolist()
    val_node_num = node_num[val_index].tolist()
    test_node_num = node_num[test_index].tolist()

    # tr_idx = data['train_idx']
    # test_id = data['test_idx']
    # t_idx = tr_idx[:int(len(tr_idx)*0.8)]
    # v_idx = tr_idx[int(len(tr_idx)*0.8)+1: int(len(tr_idx))]
    # labels = labels.reshape(labels.shape[0],1)
    data_dict['adj_list'] = adj_list
    data_dict['features_list'] = features
    data_dict['labels'] = labels
    data_dict['train_idx'] = train_index
    data_dict['val_idex'] = val_index
    data_dict['test_idx'] = test_index

    data_dict['train_node_num'] = train_node_num
    data_dict['val_node_num'] = val_node_num
    data_dict['test_node_num'] = test_node_num
    data_dict['u_all'] = data.u_all

    f_save = open(r'D:\ProjectCodes\GMExplainer\GMExplainer\data\graph_classification\causal\imdb_m.pkl', 'wb')

    pickle.dump(data_dict, f_save)

    f_save.close()




    print(1)
