import os
import numpy as np
from collections import defaultdict
import torch
from tqdm import tqdm
import dgl
import sys
import copy

from data_script import CustomGraph, STATIC_RELATIONS

sys.path.append("../")
from utils.file_process import pickle_load, pickle_dump, ensure_dir

def generator_graph_index_by_class():
    # x = pickle_load("../dataset/processed_data/test_data_index.pkl")
    # experiment = "_half"
    experiment = ""
    raw_data = pickle_load(f"../dataset/processed_data/raw_data_dict_{experiment}.pkl")
    train_data, test_data = list(), list()
    graph_node = list()
    print(raw_data.keys())
    center = raw_data['index_daily']
    label = raw_data['label']
    tqdm_bar = tqdm(center.items())

    for key, value in tqdm_bar:
        # 每一个交易日都对应一张图
        # for key, value in (tqdm_bar := tqdm(center.items())):
        # 构建节点映射
        node_map = {'center': 0}  # node的编号
        node_feature = [list(value)]  # node的特征， 来自同一个文件的特征放在一起
        node_type = [0]  # node的类型，人为定义的 :=x_type + 1
        # 构建拓扑结构
        edges = list()  # 边结构：[[x, 0]] x是node的编号
        edge_feature = list()  # [[x, 0, x_type]]
        # key:[ts_code]_[trade_date], trade_date:20211201
        for item in ['daily_basic', 'hsgt', 'shibor', 'margin', 'us_tbr', 'us_tltr', 'us_tycr', 'cn_cpi', 'cn_ppi']:
            if item in ['hsgt', "margin", 'shibor', 'us_tbr', 'us_tltr', 'us_tycr']:
                node = raw_data[item].get(key.split('_')[-1])
            elif item in ['cn_cpi', 'cn_ppi']:
                node = raw_data[item].get(key.split('_')[-1][:-2])
            else:
                node = raw_data[item].get(key)

            if node is not None:
                node_map[item] = len(node_map)
                node_feature.append(node)
                node_type.append(STATIC_RELATIONS[item] + 1)
                edges.append([node_map[item], node_map['center']])
                edge_feature.append([node_map[item], node_map['center'], STATIC_RELATIONS[item]])

        # hgst->margin
        node1 = raw_data['hsgt'].get(key.split('_')[-1])
        node2 = raw_data['margin'].get(key.split('_')[-1])
        if (node1 is not None) and (node2 is not None):
            edges.append([node_map['hsgt'], node_map['margin']])
            edge_feature.append([node_map['hsgt'], node_map['margin'], 0])

        # tycr->tbr
        node1 = raw_data['us_tycr'].get(key.split('_')[-1])
        node2 = raw_data['us_tbr'].get(key.split('_')[-1])
        if (node1 is not None) and (node2 is not None):
            edges.append([node_map['us_tycr'], node_map['us_tbr']])
            edge_feature.append([node_map['us_tycr'], node_map['us_tbr'], 0])
        # tycr->tltr
        node1 = raw_data['us_tycr'].get(key.split('_')[-1])
        node2 = raw_data['us_tltr'].get(key.split('_')[-1])
        if (node1 is not None) and (node2 is not None):
            edges.append([node_map['us_tycr'], node_map['us_tltr']])
            edge_feature.append([node_map['us_tycr'], node_map['us_tltr'], 0])
        # tbr->tltr
        node1 = raw_data['us_tbr'].get(key.split('_')[-1])
        node2 = raw_data['us_tltr'].get(key.split('_')[-1])
        if (node1 is not None) and (node2 is not None):
            edges.append([node_map['us_tbr'], node_map['us_tltr']])
            edge_feature.append([node_map['us_tbr'], node_map['us_tltr'], 0])

        # cpi->ppi
        node1 = raw_data['cn_cpi'].get(key.split('_')[-1][:-2])
        node2 = raw_data['cn_ppi'].get(key.split('_')[-1][:-2])
        if (node1 is not None) and (node2 is not None):
            edges.append([node_map['cn_cpi'], node_map['cn_ppi']])
            edge_feature.append([node_map['cn_cpi'], node_map['cn_ppi'], 0])

        # [margin, hsgt] -> [us_tycr, us_tbr, us_tltr]
        for n1 in ['margin', 'hsgt']:
            for n2 in ['us_tycr', 'us_tbr', 'us_tltr']:
                node1 = raw_data[n1].get(key.split('_')[-1][:-2])
                node2 = raw_data[n2].get(key.split('_')[-1][:-2])
                if (node1 is not None) and (node2 is not None):
                    edges.append([node_map[n1], node_map[n2]])
                    edge_feature.append([node_map[n1], node_map[n2], 0])
        # shibor -> [us_tycr, us_tltr]
        for nn in ['us_tycr', 'us_tltr']:
            node1 = raw_data['shibor'].get(key.split('_')[-1][:-2])
            node2 = raw_data[nn].get(key.split('_')[-1][:-2])
            if (node1 is not None) and (node2 is not None):
                edges.append([node_map['shibor'], node_map[nn]])
                edge_feature.append([node_map['shibor'], node_map[nn], 0])
        edges = np.array(edges)

        # 构建一个图
        mini_g = CustomGraph(topo=(edges[:, 0], edges[:, 1]))
        # 图标签
        mini_g.add_label(label[key])
        # 节点属性值
        mini_g.add_n_fea(node_feature)
        # 节点类型
        mini_g.add_n_type(node_type)
        # 边属性值
        mini_g.add_e_fea(np.array(edge_feature))
        # 子图名称（目标节点名）
        mini_g.add_name(key)
        # 将mini_g添加到dataset中
        graph_node.append(len(node_map))
        if key.split('_')[-1] >= '20210515':
            test_data.append(mini_g)
            tqdm_bar.set_description(f"#nodes: {len(node_map)}, {key.split('_')[-1]} save as test")
        else:
            train_data.append(mini_g)
            tqdm_bar.set_description(f"#nodes: {len(node_map)}, {key.split('_')[-1]} save as train")

    print(f"数据规模：Train: {len(train_data)}, Test: {len(test_data)}")
    print(f"节点规模：{sum(graph_node) / len(graph_node)}")
    pickle_dump(f"../dataset/processed_data/train_data_index{experiment}.pkl", train_data)
    pickle_dump(f"../dataset/processed_data/test_data_index{experiment}.pkl", test_data)

def generator_graph_index_by_time():
    horizon = 5
    experiment = "label_2"
    raw_data = pickle_load(f"../dataset/processed_data/raw_data_dict_{experiment}_horizon_{horizon}.pkl")
    train_data, val_data, test_data = list(), list(), list()
    graph_node = list()
    print(raw_data.keys())
    center = raw_data['pre_0']
    label = raw_data['label']
    tqdm_bar = tqdm(center.items())
    temp_data = []

    for key, value in tqdm_bar:
        # 每一个交易日都对应一张图
        # 构建节点映射
        node_map = {'pre_0': 0}  # node的编号, 记录一共有几个节点及其编号(依次排下去), {pre_0:0, pre_*:1}
        node_feature = [list(value)]  # node的特征 [[pre_0], [pre_1]]
        node_type = [0]  # 各个节点的类型，index对应node_map里的key

        # 构建拓扑结构
        edges = list()  # 边结构：[[n1, n2]] 表示n1和n2有边, 实际用map中的value代替n1 n2
        edge_feature = list()  # [[n1, n2, e_type]] 实际用map中的value代替n1 n2， e_type为边类型

        # key:[ts_code]_[trade_date], trade_date:20211201
        # 获得该交易日对应pre_1 - pre_4的信息
        pre = []
        for i in range(1, horizon):
            pre.append(f"pre_{i}")
        for item in pre:
            node = raw_data[item].get(key)
            if node is not None:
                node_map[item] = len(node_map)
                node_feature.append(node)
                node_type.append(0)  #只有一种节点类型
                # 若是其中之一，则建立边关系
                if item == "pre_1" or item == "pre_2" or item == "pre_3" or item == "pre_4":
                    edges.append([node_map[item], node_map['pre_0']])
                    relation = f"0{item[-1]}"
                    edge_feature.append([node_map[item], node_map['pre_0'], STATIC_RELATIONS[relation]])
            else:
                print(f"None Node in {key}_{item}")

        node1 = raw_data["pre_1"].get(key)
        for n in ["pre_2", "pre_3", "pre_4"]:
            node2 = raw_data[n].get(key)
            if (node1 is not None) and (node2 is not None):
                edges.append([node_map[n], node_map['pre_1']])
                relation = f"1{n[-1]}"
                edge_feature.append([node_map[n], node_map['pre_1'], STATIC_RELATIONS[relation]])

        node1 = raw_data["pre_2"].get(key)
        for n in ["pre_3", "pre_4"]:
            node2 = raw_data[n].get(key)
            if (node1 is not None) and (node2 is not None):
                edges.append([node_map[n], node_map['pre_2']])
                relation = f"2{n[-1]}"
                edge_feature.append([node_map[n], node_map['pre_2'], STATIC_RELATIONS[relation]])

        node1 = raw_data["pre_3"].get(key)
        for n in ["pre_4"]:
            node2 = raw_data[n].get(key)
            if (node1 is not None) and (node2 is not None):
                edges.append([node_map[n], node_map['pre_3']])
                relation = f"3{n[-1]}"
                edge_feature.append([node_map[n], node_map['pre_3'], STATIC_RELATIONS[relation]])

        edges = np.array(edges)

        # 构建一个图
        mini_g = CustomGraph(topo=(edges[:, 0], edges[:, 1]))
        # 图标签
        mini_g.add_label(label[key])
        # 节点属性值
        mini_g.add_n_fea(node_feature)
        # 节点类型
        mini_g.add_n_type(node_type)
        # 边属性值
        mini_g.add_e_fea(np.array(edge_feature))
        # 子图名称（目标节点名）
        mini_g.add_name(key)
        # 将mini_g添加到dataset中
        graph_node.append(len(node_map))
        if key.split('_')[-1] >= '20220415':
            test_data.append(mini_g)
            tqdm_bar.set_description(f"#nodes: {len(node_map)}, {key.split('_')[-1]} save as test")
            temp_data.append(mini_g)
        elif key.split('_')[-1] >= '20220101':
            temp_data.append(mini_g)
            val_data.append(mini_g)
        elif key.split('_')[-1] >= '20210915':
            val_data.append(mini_g)
            tqdm_bar.set_description(f"#nodes: {len(node_map)}, {key.split('_')[-1]} save as val")
        else:
            train_data.append(mini_g)
            tqdm_bar.set_description(f"#nodes: {len(node_map)}, {key.split('_')[-1]} save as train")

    print(f"数据规模：Train: {len(train_data)},  Val: {len(val_data)}, Test: {len(test_data)}, Temp: {len(temp_data)}")  # 几张图
    print(f"节点规模：{sum(graph_node) / len(graph_node)}")  # 平均每个图的节点数
    pickle_dump(f"../dataset/processed_data/train_data_index_{experiment}_horizon_{horizon}.pkl", train_data)
    pickle_dump(f"../dataset/processed_data/val_data_index_{experiment}_horizon_{horizon}.pkl", val_data)
    pickle_dump(f"../dataset/processed_data/test_data_index_{experiment}_horizon_{horizon}.pkl", test_data)
    pickle_dump(f"../dataset/processed_data/temp_data_index_{experiment}_horizon_{horizon}.pkl", temp_data)


if __name__ == '__main__':
    generator_graph_index_by_time()
    #raw_data = pickle_load(f"../dataset/processed_data/raw_data_dict_label_2_horizon_5.pkl")
    #print(len(raw_data['label']))