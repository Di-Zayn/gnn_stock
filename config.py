
class Config(object):
    def __init__(self):
        # 数据集相关配置
        self.filepath = f"./dataset/processed_data/experiment/processed_data"
        self.ds = "index_label_8_horizon_5"
        self.save_model = "experiment"
        self.fea_count = 4  # 采用几种属性
        self.train_filename = f"corr_processed_set1_data_{self.ds}.pkl"
        self.test_filename = f"corr_processed_set2_data_{self.ds}.pkl"
        self.label_log = True  # 标签是否需要取log
        # 模型超参数相关
        self.emb_dim = 100 #原始100
        self.hidden_dim = 120 # 原始120
        self.num_gnn_layer = 4 # 原始4
        self.num_heads = 3 # 原始3
        self.use_property_type = 'attn'  # [None, 'avg', 'weight', 'attn', 'bilinear']
        self.agg_mode = "center"  # ['center', 'graph', 'mixed', 'gated', 'added', 'comp']
        self.pooling_type = "sum"  # ['sum', 'avg'/'mean', 'sort', 'weight', 'gap', 's2s', 'std', 'ste']
        self.experiment = f'hgt-{self.use_property_type}'
        self.only_pos = False # 注意
        # 训练超参数相关配置
        self.seed = 619
        self.gpu_id = 1
        self.epoch = 12 # 500
        self.patient = 15
        self.batch_size = 512#128 # 256
        self.dropout = 0 #0.1
        self.lr = 1e-3 #1e-2  # 1e-3, 2e-4
        self.weight_decay = 0#0.9#0.001


