import os

import dgl
import numpy as np
import torch
from torch.utils.data import Dataset

from data_process.data_script import RELATION_NUM
from utils.file_process import pickle_load


class BaseSet(Dataset):
    def __init__(self, filepath='./processed_data', filename='processed_train_data.pkl', label_log=False):  # 初始化

        super(BaseSet, self).__init__()
        self.filepath = filepath
        self.filename = filename
        self.label_log = label_log
        self.data, self.graph, self.label, self.raw_label = None, None, None, None

    def __getitem__(self, item):
        return self.graph[item], self.label[item], self.raw_label[item]

    def __len__(self):
        return self.label.__len__()
        # return 100

    def re_normalizing_label(self, label_scale):
        print(f"重新归一化标签，use_log={self.label_log}")
        min_l, max_l = label_scale
        t_label = np.array(self.label)
        t_label = t_label * (max_l - min_l) + min_l
        raw_label = t_label.copy()
        # 值域 (0, 1)
        t_label = (t_label - min_l + 1) / (max_l - min_l + 2)

        if self.label_log:  # 值域 (0, 1)
            t_label = np.log(t_label) / np.log(1.0 / (max_l - min_l + 3))

        return raw_label.tolist(), t_label.tolist()

    def load_data(self, label_scale=None):
        if isinstance(self.filename, list):
            for n in self.filename:
                file_pathname = os.path.join(self.filepath, n)
                data_info = pickle_load(file_pathname)
        else :
            file_pathname = os.path.join(self.filepath, self.filename)
            data_info = pickle_load(file_pathname)
        self.graph = data_info["graph"]
        self.label = (np.array(data_info["label"]) > 0).astype(np.int64).tolist()
        # 之后可以调节成 多涨一点才认为是正的
        # self.raw_label, self.label = self.re_normalizing_label(label_scale)
        self.raw_label = data_info["label"]

        return self.label

    @staticmethod
    def collect_fn_graph(batch):
        """
        每一个实例构建属性星型图
        :param batch:
        :return:
        """

        # 使用图信息
        graphs, _, _ = map(list, zip(*batch))
        # graphs = [x[0] for x in batch]
        batched_graph = dgl.batch(graphs)

        # 这里可以再对图的结构做一些调整，比方说所有node_type/edge_type都弄成一样的...
        batched_graph.edata["edge_feature"] = batched_graph.edata["edge_feature"] + RELATION_NUM['MAX']  # 将边类别映射到[0~...]
        batched_graph.ndata["u_node_type"] = batched_graph.ndata.pop("node_type")

        label = [x[1] for x in batch]
        raw_label = [x[2] for x in batch]

        batch_size = len(batch)

        node_fea = batched_graph.ndata.pop('node_fea')
        node_masks = batched_graph.ndata.pop('mask_idx')

        t_label = torch.from_numpy(np.array(label)).to(torch.float)
        t_raw_label = torch.from_numpy(np.array(raw_label)).to(torch.float)

        return batched_graph, (node_fea, t_label, node_masks), t_raw_label

if __name__ == "__main__":
    ds = "index_label_2_horizon_5_2020"
    filename = [f"corr_processed_train_data_{ds}.pkl", f"corr_processed_val_data_{ds}.pkl"]
    filepath = "./dataset/processed_data/v2_feature"
    # data = BaseSet(filepath="./dataset/processed_data/v2_feature",
    #                filename=[f"corr_processed_train_data_{ds}.pkl", f"corr_processed_val_data_{ds}.pkl"], label_log=True)
    data_info = {'graph':[], 'label':[], 'name':[]}
    for n in filename:
        file_pathname = os.path.join(filepath, n)
        temp_info = pickle_load(file_pathname)
        for i in temp_info.keys():
            data_info[i].extend(temp_info[i])
    for i in data_info.keys():
        print(len(data_info[i]))