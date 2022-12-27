import os
import numpy as np
import dgl
import datetime
from collections import defaultdict, Counter
from tqdm import tqdm
import re
import inspect

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
#     "12": 1,
# }

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
    #"MAX": 4, # 不考虑正反向和自环 实际训练时(main.py) 若考虑则 relation_num = 2 * RELATION_NUM['MAX'] + 1
    "MAX": 4, #6, 2
}
FEATURE_LEN = {
    "MAX": 21 #v3：13 #v2：21 #v1：17
}





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
