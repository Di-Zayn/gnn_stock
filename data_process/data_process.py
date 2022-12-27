import numpy as np
from scipy import sparse
from tqdm import tqdm
import sys
import dgl
import torch

sys.path.append("../")
from data_script import FEATURE_LEN
from utils.file_process import pickle_load, pickle_dump


def torch_sparse_tensor_to_sparse_mx(_torch_sparse):
    """Convert a torch sparse tensor to a scipy sparse matrix."""

    m_index = _torch_sparse.coalesce().indices().numpy()
    row = m_index[0]
    col = m_index[1]
    data = _torch_sparse.coalesce().values().numpy()

    sp_matrix = sparse.coo_matrix((data, (row, col)), shape=(_torch_sparse.size()[0], _torch_sparse.size()[1]))

    return sp_matrix


def gen_nn_data(data):

    rt_data, rt_label, rt_name = list(), list(), list()

    for mini_data in tqdm(data):  # 循环数据集中的各个子图
        feature_value, feature_mask, typing_global_value = list(), list(), list()
        edge_feature = mini_data.e_fea
        node_type = mini_data.n_type
        # raw_nodes = edge_feature.shape[0]

        # adjacency_matrix()返回邻接矩阵，但返回那些1的元素,包括行列索引和weight（1）
        topology = torch_sparse_tensor_to_sparse_mx(mini_data.graph.adjacency_matrix())

        # 由于不同节点的特征长度不同，故统一尺寸，然后用mask来记录有效长度
        for node_id, node_f in enumerate(mini_data.n_fea):  # 循环子图中的结点
            mini_feature_value = [0] * FEATURE_LEN["MAX"]
            mini_feature_mask = [0] * FEATURE_LEN["MAX"]

            mini_feature_value[:len(node_f)] = node_f
            mini_feature_mask[:len(node_f)] = [1] * len(node_f)
            feature_value.append(mini_feature_value)
            feature_mask.append(mini_feature_mask)
        rt_data.append([feature_value, feature_mask, topology, edge_feature, node_type])
        rt_label.append(mini_data.delta)
        rt_name.append(mini_data.g_name)

    print(f"Raw Graph Counts:{len(data)}, Effective Graphs Count: {len(rt_label)}...")
    return rt_data, rt_label, rt_name


def gen_tensor_data(data):
    rt_data = list()
    # assert len(data) == len(label)
    for idx in tqdm(range(len(data))):
        # 获取原始数据
        feature_value, feature_mask, topology, edge_feature, node_type = data[idx]
        # o_label, c_label = label[idx]

        # 构建拓扑结构
        # graph = dgl.DGLHeteroGraph(topology)
        # #根据scipy矩阵生成图，并将weight作为edge_feature存到edata[edge_feature],所以一开始是全1
        graph = dgl.from_scipy(topology, eweight_name="edge_feature")
        # 有向图/无向图/边反向？注意！！！
        graph = dgl.to_bidirected(graph, copy_ndata=True)  # 转化为无向图
        graph = dgl.add_self_loop(graph)  # 添加自相关的边
        n_nodes = graph.number_of_nodes()

        # 一开始edge_feature全设置为0, 后续再赋值 0代表自环
        graph.edata["edge_feature"] = torch.LongTensor(graph.number_of_edges()).fill_(0)

        # 构建结点特征tensor
        node_fea = torch.FloatTensor(n_nodes, len(feature_value[0])).fill_(0)
        mask_ids = torch.ByteTensor(n_nodes, len(feature_value[0])).fill_(0)

        # graph中每一个node设置特征
        for ii in range(n_nodes):
            node_fea[ii] = torch.FloatTensor(feature_value[ii])
            mask_ids[ii] = torch.ByteTensor(feature_mask[ii])
        # 结点特征放入dgl中
        graph.ndata["node_fea"] = node_fea
        graph.ndata["mask_idx"] = mask_ids
        graph.ndata["node_type"] = torch.from_numpy(np.array(node_type))
        edge_feature = torch.from_numpy(np.array(edge_feature))
        for item in edge_feature:  # 时间慢在这个循环
            # 有向图/无向图/边反向？注意！！！
            # 必须是一+维张量，数据类型为float32，否则报错
            # unsqueeze将scale变为[]
            graph.edges[item[0], item[1]].data["edge_feature"] = item[2].unsqueeze(-1)  # 正向边
            graph.edges[item[1], item[0]].data["edge_feature"] = -item[2].unsqueeze(-1)  # 反向边

        # assert graph.number_of_edges() == len(edge_feature)  # 不能assert，因为存在多关系结点

        rt_data.append(graph)

    return rt_data

if __name__ == '__main__':
    for experiment in ["label_7", "label_8"]:
        horizon = 5
        for item in ["set1", "set2", "set3", "set4", "set5"]:
            source_data = pickle_load(f"../dataset/processed_data/experiment/processed_data/{item}_data_index_{experiment}_horizon_{horizon}.pkl")
            all_data, all_label, all_name = gen_nn_data(source_data)
            all_graph = gen_tensor_data(all_data)
            pickle_dump(f"../dataset/processed_data/experiment/processed_data/corr_processed_{item}_data_index_{experiment}_horizon_{horizon}.pkl",
                        {"graph": all_graph, "label": all_label, "name": all_name})
    # for item in ["train", "val"]:
    #     source_data = pickle_load(
    #         f"../dataset/processed_data/experiment/processed_data/{item}_data_index_{experiment}_horizon_{horizon}.pkl")
    #     all_data, all_label, all_name = gen_nn_data(source_data)
    #     all_graph = gen_tensor_data(all_data)
    #     pickle_dump(
    #         f"../dataset/processed_data/experiment/processed_data/corr_processed_{item}_data_index_{experiment}_horizon_{horizon}.pkl",
    #         {"graph": all_graph, "label": all_label, "name": all_name})



