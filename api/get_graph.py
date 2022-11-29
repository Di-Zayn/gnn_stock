from data_process.data_script import CustomGraph, FEATURE_LEN, STATIC_RELATIONS
from modules.base_model import BaseModule, NodeEmbeddingLayer, GraphAggLayer
from torch import nn as nn
from dgl.nn.pytorch import HGTConv
from torch.nn import functional as F
import dgl
import numpy as np
import torch
from scipy import sparse

class HGTModule(BaseModule):
    def __init__(self, emb_dim, hidden_dim, relation_num, node_num=1, batch_size=32, dropout=0.2,
                 num_heads=1, num_layers=2, agg_mode='graph', pooling_type=None):
        super(HGTModule, self).__init__(emb_dim, hidden_dim, batch_size, dropout)

        self.num_heads = num_heads
        self.num_layers = num_layers
        self.relation_num = relation_num
        self.node_embedding_layer = NodeEmbeddingLayer(emb_dim=emb_dim, hidden_dim=hidden_dim, node_num=node_num,
                                                       batch_size=batch_size, dropout=dropout)

        self.hgat_layer = nn.ModuleList(
            [HGTConv(emb_dim, hidden_dim, num_heads, node_num, relation_num)])
        self.batch_norm = nn.ModuleList([nn.BatchNorm1d(hidden_dim * num_heads)])
        for i in range(num_layers - 1):
            self.hgat_layer.append(
                HGTConv(hidden_dim * num_heads, hidden_dim, num_heads, node_num, relation_num))
            self.batch_norm.append(nn.BatchNorm1d(hidden_dim * num_heads))

        self.graph_agg = GraphAggLayer(in_feat=hidden_dim * num_heads, mode=agg_mode, pooling_type=pooling_type)
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim * num_heads, 2), nn.Softmax(dim=-1))

    def forward(self, batched_graph: dgl.DGLHeteroGraph, node_fea, masks):
        # 所有边的类型，所有点的类型
        edge_types = batched_graph.edata.pop("edge_feature")
        node_types = batched_graph.ndata.pop("u_node_type")

        # 节点编码 feature_size -> embedding_size
        # [node_num, embedding_size]
        embedded_nodes = self.node_embedding_layer.node_embedding(node_fea, node_types, masks)

        # 节点级信息传播
        hidden_nodes = self.hgat_layer[0](batched_graph, embedded_nodes, node_types, edge_types)
        hidden_nodes = F.relu(self.batch_norm[0](hidden_nodes))
        for i in range(1, self.num_layers):
            hidden_nodes = self.hgat_layer[i](batched_graph, hidden_nodes, node_types, edge_types)
            hidden_nodes = F.relu(self.batch_norm[i](hidden_nodes))
        batched_graph.ndata["hidden"] = hidden_nodes  # [node_num, hidden_dim * head]

        graph_emb = self.graph_agg(batched_graph, feat="hidden")  # [batch_num, hidden_dim * head]
        logits = self.output_layer(graph_emb)  # [batch_num, 2]

        return logits

# 抽时间可以把这几个函数改一改 直接用之前文件里的数据...

def get_graph(df):
    # df是按照时间顺序排列的，index是日期
    # 需要倒过来，把最近的记录放在第一位。 方便下方赋值
    feature = np.array(df.iloc[:, :].copy().sort_index(ascending=False))
    # 这里的归一化参数是否应该根据训练集来？
    feature = (feature - np.min(feature, axis=0) + 1) / (np.max(feature, axis=0) - np.min(feature, axis=0) + 2)
    raw_data = {}
    for i in range(df.shape[0]):
        raw_data[f'pre_{i}'] = feature[i]

    center = 'pre_0'
    node_map = {'pre_0': 0}  # node的编号, 记录一共有几个节点及其编号(依次排下去), {pre_0:0, pre_*:1}
    node_feature = [raw_data[center]]  # node的特征 [[pre_0], [pre_1]]
    node_type = [0]  # 各个节点的类型，index对应node_map里的key

    # 构建拓扑结构
    edges = list()  # 边结构：[[n1, n2]] 表示n1和n2有边, 实际用map中的value代替n1 n2
    edge_feature = list()  # [[n1, n2, e_type]] 实际用map中的value代替n1 n2， e_type为边类型

    pre = []
    for i in range(1, 5):
        pre.append(f"pre_{i}")
    for item in pre:
        node = raw_data[item]
        if node is not None:
            node_map[item] = len(node_map)
            node_feature.append(node)
            node_type.append(0)  # 只有一种节点类型
            edges.append([node_map[item], node_map['pre_0']])
            relation = f"0{item[-1]}"
            edge_feature.append([node_map[item], node_map['pre_0'], STATIC_RELATIONS[relation]])
        else:
            print(f"None Node in {item}")

    node1 = raw_data["pre_1"]
    for n in ["pre_2", "pre_3", "pre_4"]:
        node2 = raw_data[n]
        if (node1 is not None) and (node2 is not None):
            edges.append([node_map[n], node_map['pre_1']])
            relation = f"1{n[-1]}"
            edge_feature.append([node_map[n], node_map['pre_1'], STATIC_RELATIONS[relation]])

    node1 = raw_data["pre_2"]
    for n in ["pre_3", "pre_4"]:
        node2 = raw_data[n]
        if (node1 is not None) and (node2 is not None):
            edges.append([node_map[n], node_map['pre_2']])
            relation = f"2{n[-1]}"
            edge_feature.append([node_map[n], node_map['pre_2'], STATIC_RELATIONS[relation]])

    node1 = raw_data["pre_3"]
    for n in ["pre_4"]:
        node2 = raw_data[n]
        if (node1 is not None) and (node2 is not None):
            edges.append([node_map[n], node_map['pre_3']])
            relation = f"3{n[-1]}"
            edge_feature.append([node_map[n], node_map['pre_3'], STATIC_RELATIONS[relation]])

    edges = np.array(edges)

    # 构建一个图
    mini_g = CustomGraph(topo=(edges[:, 0], edges[:, 1]))
    # 节点属性值
    mini_g.add_n_fea(node_feature)
    # 节点类型
    mini_g.add_n_type(np.array(node_type, dtype=np.int64))
    # 边属性值
    mini_g.add_e_fea(np.array(edge_feature, dtype=np.int64))
    # 子图名称（目标节点名）
    mini_g.add_name("predict")

    nn_data = gen_nn_data(mini_g)

    graph = gen_tensor_data(nn_data)
    return graph

def torch_sparse_tensor_to_sparse_mx(_torch_sparse):
    """Convert a torch sparse tensor to a scipy sparse matrix."""

    m_index = _torch_sparse.coalesce().indices().numpy()
    row = m_index[0]
    col = m_index[1]
    data = _torch_sparse.coalesce().values().numpy()

    sp_matrix = sparse.coo_matrix((data, (row, col)), shape=(_torch_sparse.size()[0], _torch_sparse.size()[1]))

    return sp_matrix


def gen_nn_data(mini_g):

    feature_value, feature_mask = list(), list()
    edge_feature = mini_g.e_fea
    node_type = mini_g.n_type

    # adjacency_matrix()返回邻接矩阵，但返回那些1的元素,包括行列索引和weight（1）
    topology = torch_sparse_tensor_to_sparse_mx(mini_g.graph.adjacency_matrix())

    # 由于不同节点的特征长度不同，故统一尺寸，然后用mask来记录有效长度
    for node_id, node_f in enumerate(mini_g.n_fea):  # 循环子图中的结点
        mini_feature_value = [0] * FEATURE_LEN["MAX"]
        mini_feature_mask = [0] * FEATURE_LEN["MAX"]

        mini_feature_value[:len(node_f)] = node_f
        mini_feature_mask[:len(node_f)] = [1] * len(node_f)
        feature_value.append(mini_feature_value)
        feature_mask.append(mini_feature_mask)
    rt_data = [feature_value, feature_mask, topology, edge_feature, node_type]

    return rt_data

def gen_tensor_data(data):

    # 获取原始数据
    feature_value, feature_mask, topology, edge_feature, node_type = data

    # 构建拓扑结构
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
    for item in edge_feature:
        graph.edges[item[0], item[1]].data["edge_feature"] = item[2].unsqueeze(-1)  # 正向边
        graph.edges[item[1], item[0]].data["edge_feature"] = -item[2].unsqueeze(-1)  # 反向边

    return graph