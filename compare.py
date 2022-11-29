import os
from collections import Counter

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data_process.data_script import RELATION_NUM
from dataset.nn_data_generator import BaseSet
from main import set_random_seed
from modules.graph_model import HGTModule
from utils.tools import precision_recall_fscore

config = Config()

model_save_path = f"./saved_models/model-{config.ds}-x5_seed{config.seed}.pkl"
print(model_save_path)
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
assert torch.cuda.is_available() is True
device = torch.device(torch.cuda.current_device())
set_random_seed(config.seed)  # 一定放在gpu设置后面


if config.only_pos:
    relation_num = RELATION_NUM['MAX'] + 1
else:
    relation_num = 2 * RELATION_NUM["MAX"] + 1
# relation_num = 21
print(f"relation_num:{relation_num}")
model = HGTModule(emb_dim=config.emb_dim, hidden_dim=config.hidden_dim,
                  relation_num=relation_num, node_num=1,
                  batch_size=config.batch_size, dropout=config.dropout, num_heads=config.num_heads,
                  num_layers=config.num_gnn_layer, agg_mode=config.agg_mode, pooling_type=config.pooling_type)
model.load_state_dict(torch.load(model_save_path))
model = model.to(device)

for p in ["temp_data_index_label_2_horizon_5"]:
    test_filename = f"corr_processed_{p}.pkl"
    test_dataset = BaseSet(filepath=config.filepath, filename=test_filename, label_log=config.label_log)
    test_label = test_dataset.load_data()
    print(f'Data Load Success test_data:{len(test_dataset.label)}')
    print("Test:", Counter(test_label))
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                             collate_fn=test_dataset.collect_fn_graph, num_workers=4)

    dev_loss = 0
    true_tags, pred_tags, raw_tags = list(), list(), list()
    model.eval()
    with torch.no_grad():
        for batch_idx, (b_graph, b_data, b_raw_label) in enumerate(tqdm(test_loader)):
            # b_graph, b_data = b_source
            b_graph = b_graph.to(device)
            node_fea, l_label, mask_ids = b_data
            raw_label = b_raw_label  # 原始label，偏移量
            node_fea = node_fea.to(device)
            l_label = l_label.to(torch.int64).to(device)
            mask_ids = mask_ids.to(device)

            logits, batch_dev_loss = model.forward(b_graph, node_fea, mask_ids, l_label)

            true_tags.extend(l_label.detach().cpu().tolist())
            raw_tags.extend(raw_label.detach().cpu().tolist())

            pred_tags.extend(torch.argmax(logits, dim=-1).to(torch.int64).detach().cpu().tolist())
            dev_loss += batch_dev_loss

        print(f"dev_loss: {dev_loss / test_dataset.__len__()}")
        print("=========== 归一化后指标 ===========")
        prf1 = precision_recall_fscore(pred_list=pred_tags, true_list=true_tags)
        print(Counter(pred_tags), Counter(true_tags))
        print("=========== 结束 ===========")
