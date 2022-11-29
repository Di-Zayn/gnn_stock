import os
import numpy as np
import dgl
import datetime
from collections import defaultdict, Counter
from tqdm import tqdm
import re
import inspect

# STATIC_RELATIONS = {
#     "moneyflow": 0,
#     "margin_detail": 1,
#     "pledge_detail": 2,
#     "hsgt": 3,
#     "margin": 4
# }
# ['daily_basic', 'margin', 'hsgt', 'shibor', 'us_tbr', 'us_tltr', 'us_tycr', 'cn_cpi', 'cn_ppi']
# STATIC_RELATIONS = {
#     "daily_basic": 0,
#     "margin": 1,
#     "hsgt": 2,
#     "shibor": 3,
#     "us_tbr": 4,
#     "us_tltr": 5,
#     "us_tycr": 6,
#     "cn_cpi": 7,
#     "cn_ppi": 8,
# }
# STATIC_RELATIONS = {
#     "01": 1,
#     "02": 2,
#     "12": 1,
# }
STATIC_RELATIONS = {
    "01": 1,
    "02": 2,
    "03": 3,
    "04": 4,
    "12": 1,
    "13": 2,
    "14": 3,
    "23": 1,
    "24": 2,
    "34": 1,
}
# STATIC_RELATIONS = {
#     "01": 1,
#     "02": 2,
#     "03": 3,
#     "04": 4,
#     "05": 5,
#     "06": 6,
#     "12": 1,
#     "13": 2,
#     "14": 3,
#     "15": 4,
#     "16": 5,
#     "23": 1,
#     "24": 2,
#     "25": 3,
#     "26": 4,
#     "34": 1,
#     "35": 2,
#     "36": 3,
#     "45": 1,
#     "46": 2,
#     "56": 1
# }
RELATION_NUM = {
    "MAX": 4 #2 #6  # 不考虑正反向和自环
}
FEATURE_LEN = {
    "MAX": 21 # 17
}

# FEATURE_LEN = {
#     "MAX": 18,
#     "center": 10,
#     "moneyflow": 18,
#     "margin_detail": 8,
#     "pledge_detail": 5,
#     "hsgt": 6,
#     "margin": 7
# }
# FEATURE_LEN = {
#     "MAX": 30,
#     "center": 10,
#     "daily_basic": 10,
#     "margin": 7,
#     "hsgt": 6,
#     "shibor": 8,
#     "us_tbr": 10,
#     "us_tltr": 2,
#     "us_tycr": 12,
#     "cn_cpi": 12,
#     "cn_ppi": 30,
# }

# FEATURE_LEN = {
#     "name": ["center", "moneyflow", "margin_detail", "pledge_detail", "hsgt", "margin"],
#     "length": [10, 18, 8, 5, 6, 7]
# }


def retrieve_name(var):
    """
    Gets the name of var. Does it from the out most frame inner-wards.
    :param var: variable to get name from.
    :return: string
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


class CustomGraph(object):
    def __init__(self, topo):
        self.graph: dgl.DGLHeteroGraph = dgl.graph(topo)
        self.n_fea = None
        self.n_type = None
        self.e_fea = None
        self.label, self.delta = None, None
        self.g_name = ""

    def add_label(self, _time_delta):
        """
        构建图的标签
        :return:
        """
        self.delta = _time_delta
        # 实际没使用
        self.label = 1 if _time_delta >= 0 else 0

    def add_name(self, g_name: str):
        self.g_name = g_name

    def add_e_fea(self, e_fea: np.array):
        assert e_fea.shape[0] == self.graph.number_of_edges()
        self.e_fea = e_fea

    def add_n_fea(self, n_fea: list):
        assert n_fea.__len__() == self.graph.number_of_nodes()
        self.n_fea = n_fea

    def add_n_type(self, n_type: list):
        assert n_type.__len__() == self.graph.number_of_nodes()
        self.n_type = n_type
