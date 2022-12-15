import time

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
import random
import dgl
from tqdm import tqdm
import os
from datetime import datetime
from collections import Counter

from config import Config
from dataset.nn_data_generator import BaseSet
from data_process.data_script import STATIC_RELATIONS, RELATION_NUM
from modules.graph_model import HGTModule
from utils.file_process import pickle_load, pickle_dump, ensure_dir
from utils.tools import regression_metric, precision_recall_fscore




def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dgl.seed(seed)


# 是一下把label改成与前五日最小值的变化  可以试试
# 或者在loss里增加权重惩罚 先试试这
# 调一下label1 调参数
# 以及label1—all-connect
# 每日都变化的图结构

if __name__ == "__main__":
    print("Sleep...")
    time.sleep(0)
    print("Begin...")
    config = Config()
    ensure_dir("./saved_models")
    ensure_dir("./saved_models/v2_feature")
    ensure_dir("./results")
    ensure_dir("./results/test")

    model_save_path = f"./saved_models/v2_feature/model-{config.ds}-x5_seed{config.seed}.pkl"
    result_save_path = f"./results/model-{config.ds}-x5.pkl"
    result_save_path2 = f"./results/test/model-{config.ds}-x5.pkl"
    print(model_save_path)
    print(result_save_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    assert torch.cuda.is_available() is True
    device = torch.device(torch.cuda.current_device())

    set_random_seed(config.seed)  # 一定放在gpu设置后面

    start_time = datetime.now()

    train_dataset = BaseSet(filepath=config.filepath, filename=config.train_filename, label_log=config.label_log)
    train_label = train_dataset.load_data()

    test_dataset = BaseSet(filepath=config.filepath, filename=config.test_filename, label_log=config.label_log)
    test_label = test_dataset.load_data()
    print(f'Data Load Success, train_data: {len(train_dataset.label)};'
          f' test_data:{len(test_dataset.label)}'
          )
    print("Train:", Counter(train_label), ";"
                                          # " Test:", Counter(test_label)
          )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              collate_fn=train_dataset.collect_fn_graph, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                             collate_fn=test_dataset.collect_fn_graph, num_workers=4)

    if config.only_pos:
        relation_num = RELATION_NUM['MAX'] + 1
    else:
        relation_num = 2 * RELATION_NUM["MAX"] + 1
    print(f"relation_num:{relation_num}")
    model = HGTModule(emb_dim=config.emb_dim, hidden_dim=config.hidden_dim,
                      relation_num= relation_num, node_num=1,
                      batch_size=config.batch_size, dropout=config.dropout, num_heads=config.num_heads,
                      num_layers=config.num_gnn_layer, agg_mode=config.agg_mode, pooling_type=config.pooling_type)
    print("模型参数量:", sum(p.numel() for p in model.parameters()))
    print("可训练参数量:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    initial_time = datetime.now()
    initial_delta = (initial_time - start_time).seconds
    print(f"初始化时间：{initial_delta // 3600}小时，{(initial_delta % 3600) // 60}分钟，{initial_delta % 60}秒")
    # 初始化结束

    print("参数:")
    print(f"lr:{config.lr}, weight_decay:{config.weight_decay}, drop_out:{config.dropout}, batch_size:{config.batch_size}")
    print("模型参数")
    print(f"embedding:{config.emb_dim} hidden:{config.hidden_dim} head:{config.num_heads} layer：{config.num_gnn_layer}")

    best_epoch, best_f1, current_patient = 0, 0, 0
    f1_precession, f1_recall = 0, 0

    for epoch in range(config.epoch):
        torch.cuda.empty_cache()
        total_loss = 0
        train_logit_p, train_logit_t = list(), list()

        model.train()
        print(f"Epoch: {epoch + 1}, Start...")

        # Dataset中的自定义方法会将多张图整合到一个batch中
        for batch_idx, (b_graph, b_data, _) in enumerate(tqdm(train_loader)):
            # _对应的是未映射到0/1的原始label 不用
            b_graph = b_graph.to(device)

            node_fea, l_label, mask_ids = b_data
            node_fea = node_fea.to(device)
            l_label = l_label.to(torch.int64).to(device)
            mask_ids = mask_ids.to(device)

            model.zero_grad()
            logits, loss = model.forward(b_graph, node_fea, mask_ids, l_label)
            train_logit_t.extend(l_label.detach().cpu().tolist())
            train_logit_p.extend(torch.argmax(logits, dim=-1).to(torch.int64).detach().cpu().tolist())
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch + 1}, train_loss = {total_loss / train_dataset.__len__()}")
        precision_recall_fscore(train_logit_p, train_logit_t)
        print(Counter(train_logit_p), Counter(train_logit_t))

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

                # pred_tags.extend((logits >= 0.5).to(torch.int64).detach().cpu().tolist())
                pred_tags.extend(torch.argmax(logits, dim=-1).to(torch.int64).detach().cpu().tolist())
                # model.decode()
                dev_loss += batch_dev_loss

            print(f"===>epoch {epoch + 1}, dev_loss: {dev_loss / test_dataset.__len__()}")
            print("=========== 归一化后指标 ===========")
            # epoch_r2 = regression_metric(pred_list=pred_tags, true_list=true_tags)
            prf1 = precision_recall_fscore(pred_list=pred_tags, true_list=true_tags)
            print(Counter(pred_tags), Counter(true_tags))
            epoch_f1 = prf1["micro"][-1]

            if epoch_f1 > best_f1:
                best_epoch = epoch + 1
                best_f1 = epoch_f1
                current_patient = 0
                print(f"Epoch-{epoch + 1} get better F1-score: {epoch_f1}")
                f1_precession = prf1["micro"][0]
                f1_recall = prf1["micro"][1]
                # save model
                print("Save Current Model...")
                torch.save(model.state_dict(), model_save_path)
            else:
                print(f"Current Patient: {current_patient + 1}")
                if current_patient >= config.patient:
                    end_time = datetime.now()
                    print(f"Early Stopping as Epoch-{epoch + 1}...")
                    print(f"Best Epoch: Epoch-{best_epoch}, Best F1-score={best_f1}, "
                          f"Precession={f1_precession}, Recall={f1_recall}")
                    train_delta = (end_time - initial_time).seconds
                    print(f"训练时间：{train_delta // 3600}小时，{(train_delta % 3600) // 60}分钟，{train_delta % 60}秒")
                    break
                current_patient += 1
    print("Finished.......")