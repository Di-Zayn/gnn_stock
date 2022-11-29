import dgl
import numpy as np
from dgl.nn.pytorch import AvgPooling, SumPooling, MaxPooling, SortPooling, WeightAndSum, Set2Set
from dgl.nn.pytorch import GlobalAttentionPooling, SetTransformerEncoder, SetTransformerDecoder
from dgl import function
from dgl.utils import expand_as_pair
import torch
import torch.nn as nn
from torch.nn import functional as F
# from modules.aggregation_model import GraphAggLayer

import sys

from config import Config

sys.path.append("../")
from data_process.data_script import STATIC_RELATIONS, FEATURE_LEN


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""

    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, bias=True):
        """MLP layers construction

        Parameters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction

        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim, bias=bias))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            self.linears.append(nn.Linear(hidden_dim, output_dim, bias=bias))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class BaseModule(nn.Module):
    def __init__(self, emb_dim, hidden_dim, batch_size=32, dropout=0.2):
        super(BaseModule, self).__init__()
        self.embedding_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.dropout = dropout
        self.hidden_fc = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(emb_dim, hidden_dim))  # 激活？tanh/ReLU，注意！！！
        # self.output_layer = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim, 2), nn.Softmax(dim=-1))
        # self.loss_func = nn.MSELoss(reduction='sum')
        # self.loss_func = nn.BCELoss(reduction='sum')
        config = Config()
        if config.loss_func_type == "label_2":
            print(f"loss_func_type:{config.loss_func_type}")
            self.loss_func = nn.CrossEntropyLoss(reduction='sum', weight=torch.from_numpy(np.array([1, 6])).float())
        elif config.loss_func_type == "label_1":
            print(f"loss_func_type:{config.loss_func_type}")
            self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        elif config.loss_func_type == "label_3":
            print(f"loss_func_type:{config.loss_func_type}")
            self.loss_func = nn.CrossEntropyLoss(reduction='sum', weight=torch.from_numpy(np.array([1, 2])).float())

    def calc_loss(self, logits, true_prob):
        return self.loss_func(logits, true_prob)

    def forward(self, *x):
        raise NotImplementedError


class NodeEmbeddingLayer(BaseModule):
    def __init__(self, emb_dim, hidden_dim, node_num, batch_size=32, dropout=0.2):
        super(NodeEmbeddingLayer, self).__init__(emb_dim, hidden_dim, batch_size, dropout)

        # 有多层,每一层处理一种类型的node
        self.feature_embedding_layer = nn.ModuleList(
            [nn.Linear(FEATURE_LEN["MAX"], emb_dim) for _ in range(node_num)])
        # len(STATIC_RELATIONS) + 1)

    def node_embedding(self, node_fea, node_type, masks):
        """
        :node_fea: features of each vertex
        :node_type: type of each vertex
        :masks: masks of each feature
        """
        # masks = masks.unsqueeze(-1)
        # 属性编码
        node_emb = torch.zeros_like(self.feature_embedding_layer[0](node_fea))
        for idx, nt in enumerate(node_type):
            node_emb[idx] = self.feature_embedding_layer[nt](node_fea[idx] * masks[idx])
        # mask聚合
        # node_emb = node_emb
        # property_emb = torch.sum(property_emb, dim=1) / (torch.sum(masks, dim=1) + (torch.sum(masks, 1) == 0))
        return node_emb

    def forward(self, graph, node_fea, masks, label):
        # 这个应该没用过...
        print("NodeEmbeddingForward")
        node_types = graph.ndata.pop("u_node_type")
        property_emb = self.node_embedding(node_fea, node_types, masks)

        graph_emb = self.hidden_fc(property_emb)
        graph.ndata['h'] = graph_emb

        g_node_id = torch.zeros_like(graph.batch_num_nodes())
        for i in range(g_node_id.shape[0]):
            g_node_id[i] = torch.sum(graph.batch_num_nodes()[:i])
        graph_emb = graph.ndata['h'][g_node_id]  # 包含拓扑结构的中心节点

        # graph_emb = dgl.mean_nodes(graph, 'h')  # 平均所有节点
        logits = self.output_layer(graph_emb)
        # logits = logits.squeeze()

        return logits, self.loss_func(logits, label)


class GraphAggLayer(nn.Module):
    def __init__(self, in_feat: int = None, mode='center', pooling_type=None):
        """
        :param in_feat: 特征维度
        :param mode: 图的使用形式，['center', 'graph']，只使用中心节点/使用图结构
        :param pooling_type: 池化方式，['sum', 'avg', ...]，sum/mean/max/sort等
        """
        super(GraphAggLayer, self).__init__()
        assert mode in ['center', 'graph', 'mixed', 'gated', 'added', 'comp']
        self.in_feat = in_feat
        self.mode = mode
        if mode == 'center':
            return

        self.pooling_type = pooling_type
        self.pool_num = 2
        if pooling_type == 'mean' or pooling_type == 'avg':
            self.pooling = AvgPooling()
        elif pooling_type == 'sum':
            self.pooling = SumPooling()
        elif pooling_type == 'sort':
            self.pooling = SortPooling(k=self.pool_num)
        elif pooling_type == 'weight':
            self.pooling = WeightAndSum(self.in_feat)
        elif pooling_type == 'gap':  # global attention pooling
            gate_nn = nn.Linear(self.in_feat, 1)
            feat_nn = nn.Linear(self.in_feat, self.in_feat)
            self.pooling = GlobalAttentionPooling(gate_nn, feat_nn)
        elif pooling_type == 's2s':
            self.pooling = Set2Set(self.in_feat, self.pool_num, n_layers=1)
            # self.pooling_after = nn.Linear(self.in_feat * 2, self.in_feat)
        elif pooling_type == 'ste' or pooling_type == 'std':
            self.pooling = SetTransformerDecoder(self.in_feat, 4, 4, 20, 1, self.pool_num)
        else:
            self.pooling = None
            if self.mode == 'graph':
                raise NotImplementedError

        if pooling_type == 'ste':
            self.pooling_before = SetTransformerEncoder(self.in_feat, 4, 4, 20)

        if mode == 'mixed' or mode == 'comp':
            self.out_fc = nn.Linear(in_feat * 2, in_feat)
        if mode == 'gated':
            self.center_fc = nn.Sequential(nn.Linear(in_feat, in_feat), nn.BatchNorm1d(in_feat), nn.ReLU())
            # self.out_gate = nn.Parameter(torch.empty(2, in_feat))
            self.out_gate = nn.Parameter(torch.empty(1, in_feat))
            # nn.init.xavier_uniform_(self.out_gate)
            nn.init.xavier_normal_(self.out_gate)

    def forward(self, b_graph, feat: str):
        # 图级聚合
        if self.mode == 'center':  # 1.取中心节点
            # b_graph.batch_num_nodes() 返回batch里各子图的节点数量, [batch_size]
            g_node_id = torch.zeros_like(b_graph.batch_num_nodes())
            for i in range(g_node_id.shape[0]):
                # 第i张图前的节点数
                g_node_id[i] = torch.sum(b_graph.batch_num_nodes()[:i])
            # 得到每张图的第一个节点的信息,而第一个节点是中心节点
            graph_emb = b_graph.ndata[feat][g_node_id]  # 包含拓扑结构的中心节点
        elif self.mode == 'graph' or self.mode == 'mixed' or self.mode == 'gated' or self.mode == 'added' \
                or self.mode == 'comp':
            # 2.平均所有结点
            if self.pooling_type == 'ste':
                graph_emb = self.pooling_before(b_graph, b_graph.ndata[feat])
                graph_emb = self.pooling(b_graph, graph_emb)
            else:
                graph_emb = self.pooling(b_graph, b_graph.ndata[feat])

            if self.pooling_type in ['sort', 'ste', 'std', 's2s']:
                graph_emb = graph_emb.reshape(graph_emb.shape[0], self.pool_num, -1)
                graph_emb = torch.mean(graph_emb, dim=1)
            # if self.pooling_type == 's2s':
            #     graph_emb = self.pooling_after(graph_emb)
            if self.mode == 'mixed' or self.mode == 'gated' or self.mode == 'added' or self.mode == 'comp':
                # 3.混合中心节点&平均结点
                # 添加center特征
                g_node_id = torch.zeros_like(b_graph.batch_num_nodes())
                for i in range(g_node_id.shape[0]):
                    g_node_id[i] = torch.sum(b_graph.batch_num_nodes()[:i])
                center_emb = b_graph.ndata[feat][g_node_id]  # 包含拓扑结构的中心节点

                if self.mode == 'added':
                    graph_emb = (center_emb + graph_emb) / 2
                elif self.mode == 'mixed':
                    graph_emb = torch.cat([graph_emb, center_emb], dim=-1)
                    graph_emb = self.out_fc(graph_emb)
                elif self.mode == 'comp':
                    graph_emb = torch.cat([(graph_emb + center_emb), (graph_emb * center_emb)], dim=-1)
                    graph_emb = self.out_fc(graph_emb)
                elif self.mode == 'gated':
                    center_emb = self.center_fc(center_emb)
                    # 0.gate向量与embedding的相似度
                    graph_emb = torch.cat([graph_emb.unsqueeze(1), center_emb.unsqueeze(1)], dim=1)  # [B, 2, D]
                    att_score = torch.softmax(graph_emb * self.out_gate.unsqueeze(0), dim=1)
                    graph_emb = torch.sum(graph_emb * att_score, dim=1)
        else:
            raise NotImplementedError

        return graph_emb
