
class Config(object):
    def __init__(self):
        # 数据集相关配置
        self.filepath = f"./dataset/processed_data/v2_feature"
        self.ds = "index_label_2_horizon_5"
        self.only_pos = False # 注意
        self.fea_count = 4  # 采用几种属性
        self.train_filename = f"corr_processed_train_data_{self.ds}.pkl"
        self.test_filename = f"corr_processed_val_data_{self.ds}.pkl"
        self.label_log = True  # 标签是否需要取log
        # 模型超参数相关
        self.emb_dim = 100
        self.hidden_dim = 120
        self.num_gnn_layer = 4
        self.num_heads = 3
        self.use_property_type = 'attn'  # [None, 'avg', 'weight', 'attn', 'bilinear']
        self.agg_mode = "center"  # ['center', 'graph', 'mixed', 'gated', 'added', 'comp']
        self.pooling_type = "sum"  # ['sum', 'avg'/'mean', 'sort', 'weight', 'gap', 's2s', 'std', 'ste']
        self.experiment = f'hgt-{self.use_property_type}'
        self.loss_func_type = "label_2" # 注意
        # 训练超参数相关配置
        self.seed = 200
        self.gpu_id = 0
        self.epoch = 50 # 500
        self.patient = 25
        self.batch_size = 256 # 256
        self.dropout = 0.1
        self.lr = 1e-2  # 1e-3, 2e-4
        self.weight_decay = 0.001