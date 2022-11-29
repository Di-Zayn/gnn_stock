from collections import Counter

import dgl
from dgl.nn.pytorch import HGTConv, TypedLinear
import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.base_model import BaseModule, NodeEmbeddingLayer, GraphAggLayer, STATIC_RELATIONS


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

    def forward(self, batched_graph: dgl.DGLHeteroGraph, node_fea, masks, label):
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
        # logits = logits.squeeze(-1)
        return logits, self.loss_func(logits, label)
