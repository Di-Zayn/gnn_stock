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

def generator_graph_index_by_time(experiment, horizon):
    raw_data = pickle_load(f"../dataset/processed_data/experiment/processed_data/raw_data_dict_{experiment}_horizon_5.pkl")

    # train_data, val_data, test_data = list(), list(), list()
    set1, set2, set3, set4, set5 = list(), list(), list(), list(), list()

    graph_node = list()
    print(raw_data.keys())
    center = raw_data['pre_0'] # 如果图聚合要使用center模式 这里就要先把pre_0拿出来
    label = raw_data['label']
    tqdm_bar = tqdm(center.items())

    # 每一个交易日都对应一张图
    for key, value in tqdm_bar:
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
                # if item == "pre_1" or item == "pre_2" or item == "pre_3" or item == "pre_4" or item == "pre_5" or item == "pre_6":
                if item == "pre_1" or item == "pre_2" or item == "pre_3" or item == "pre_4":
                    edges.append([node_map[item], node_map['pre_0']])
                    relation = f"0{item[-1]}"
                    edge_feature.append([node_map[item], node_map['pre_0'], STATIC_RELATIONS[relation]])
            else:
                print(f"None Node in {key}_{item}")

        node1 = raw_data["pre_1"].get(key)
        # for n in ["pre_2", "pre_3", "pre_4", "pre_5", "pre_6"]:
        for n in ["pre_2", "pre_3", "pre_4"]:
            node2 = raw_data[n].get(key)
            if (node1 is not None) and (node2 is not None):
                edges.append([node_map[n], node_map['pre_1']])
                relation = f"1{n[-1]}"
                edge_feature.append([node_map[n], node_map['pre_1'], STATIC_RELATIONS[relation]])

        node1 = raw_data["pre_2"].get(key)
        for n in ["pre_3", "pre_4"]:
        # for n in ["pre_3", "pre_4", "pre_5", "pre_6"]:
            node2 = raw_data[n].get(key)
            if (node1 is not None) and (node2 is not None):
                edges.append([node_map[n], node_map['pre_2']])
                relation = f"2{n[-1]}"
                edge_feature.append([node_map[n], node_map['pre_2'], STATIC_RELATIONS[relation]])

        node1 = raw_data["pre_3"].get(key)
        for n in ["pre_4"]:
        # for n in ["pre_4", "pre_5", "pre_6"]:
            node2 = raw_data[n].get(key)
            if (node1 is not None) and (node2 is not None):
                edges.append([node_map[n], node_map['pre_3']])
                relation = f"3{n[-1]}"
                edge_feature.append([node_map[n], node_map['pre_3'], STATIC_RELATIONS[relation]])

        # node1 = raw_data["pre_4"].get(key)
        # for n in ["pre_5", "pre_6"]:
        #     node2 = raw_data[n].get(key)
        #     if (node1 is not None) and (node2 is not None):
        #         edges.append([node_map[n], node_map['pre_4']])
        #         relation = f"4{n[-1]}"
        #         edge_feature.append([node_map[n], node_map['pre_4'], STATIC_RELATIONS[relation]])
        #
        # node1 = raw_data["pre_5"].get(key)
        # for n in ["pre_6"]:
        #     node2 = raw_data[n].get(key)
        #     if (node1 is not None) and (node2 is not None):
        #         edges.append([node_map[n], node_map['pre_5']])
        #         relation = f"5{n[-1]}"
        #         edge_feature.append([node_map[n], node_map['pre_5'], STATIC_RELATIONS[relation]])

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

        if key.split('_')[-1] < '20210501':
            set1.append(mini_g)
            tqdm_bar.set_description(f"#nodes: {len(node_map)}, {key.split('_')[-1]} save as set1")
        elif key.split('_')[-1] < '20210701':
            set2.append(mini_g)
            tqdm_bar.set_description(f"#nodes: {len(node_map)}, {key.split('_')[-1]} save as set2")
        elif key.split('_')[-1] < '20210901':
            set3.append(mini_g)
            tqdm_bar.set_description(f"#nodes: {len(node_map)}, {key.split('_')[-1]} save as set3")
        elif key.split('_')[-1] < '20211101':
            set4.append(mini_g)
            tqdm_bar.set_description(f"#nodes: {len(node_map)}, {key.split('_')[-1]} save as set4")
        else:
            set5.append(mini_g)
            tqdm_bar.set_description(f"#nodes: {len(node_map)}, {key.split('_')[-1]} save as set5")

        # if key.split('_')[-1] >= '20220415':
        #     test_data.append(mini_g)
        #     tqdm_bar.set_description(f"#nodes: {len(node_map)}, {key.split('_')[-1]} save as test")
        # elif key.split('_')[-1] >= '20210915':
        #     val_data.append(mini_g)
        #     tqdm_bar.set_description(f"#nodes: {len(node_map)}, {key.split('_')[-1]} save as val")
        # else:
        #     train_data.append(mini_g)
        #     tqdm_bar.set_description(f"#nodes: {len(node_map)}, {key.split('_')[-1]} save as train")

    # print(f"数据规模：Train: {len(train_data)},  Val: {len(val_data)}, Test: {len(test_data)}")  # 几张图
    print(f"数据规模: set1:{len(set1)}, set2:{len(set2)}, set3:{len(set3)}, set4:{len(set4)}, set5:{len(set5)}")
    print(f"节点规模：{sum(graph_node) / len(graph_node)}")  # 平均每个图的节点数
    pickle_dump(f"../dataset/processed_data/experiment/processed_data/set1_data_index_{experiment}_horizon_{horizon}.pkl", set1)
    pickle_dump(f"../dataset/processed_data/experiment/processed_data/set2_data_index_{experiment}_horizon_{horizon}.pkl", set2)
    pickle_dump(f"../dataset/processed_data/experiment/processed_data/set3_data_index_{experiment}_horizon_{horizon}.pkl", set3)
    pickle_dump(f"../dataset/processed_data/experiment/processed_data/set4_data_index_{experiment}_horizon_{horizon}.pkl", set4)
    pickle_dump(f"../dataset/processed_data/experiment/processed_data/set5_data_index_{experiment}_horizon_{horizon}.pkl", set5)

    # pickle_dump(f"../dataset/processed_data/experiment/processed_data/train_data_index_{experiment}_horizon_{horizon}.pkl", train_data)
    # pickle_dump(f"../dataset/processed_data/experiment/processed_data/val_data_index_{experiment}_horizon_{horizon}.pkl", val_data)
    # pickle_dump(f"../dataset/processed_data/test_data_index_{experiment}_horizon_{horizon}.pkl", test_data)


if __name__ == '__main__':
    for e in ["label_7", "label_8"]:
        h = 5
        generator_graph_index_by_time(e, h)
