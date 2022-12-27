import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
from collections import Counter
from data_script import retrieve_name

sys.path.append("../")
from utils.file_process import ensure_dir, pickle_dump

def generator_by_diff_time(label_type, horizon):

    pre = []
    for i in range(horizon):
        df = pd.read_csv(f"../dataset/processed_data/experiment/pre_{i}.csv")
        # df = df[df['trade_date'] < 20210101]
        # df.index = range(len(df))
        pre.append(df)
    label_dict = dict()

    label = pd.read_csv(f"../dataset/processed_data/experiment/nxt_{label_type}.csv")
    # label = label[label['trade_date'] < 20210101]
    # label.index = range(len(label))

    # 将index_daily的label提取出来
    for i in range(label.shape[0]):
        # label_dict[f"{label.iloc[i, 0]}_{label.iloc[i, 1]}"] = label.iloc[i, 2]
        label_dict[f"{label['ts_code'][i]}_{label['trade_date'][i]}"] = label['label'][i]
    raw_data_all_in_one = {"label": label_dict}

    ts_code = label['ts_code'].copy()
    date = label['trade_date'].copy()

    for i in range(horizon):
        temp_dict = {}
        feature = np.array(pre[i].iloc[:, 2:].copy())
        feature = (feature - np.mean(feature, axis=0)) / np.std(feature, axis=0)
        # 论文是标准化
        # feature = (feature - np.min(feature, axis=0) + 1) / (np.max(feature, axis=0) - np.min(feature, axis=0) + 2)
        for j in range(pre[i].shape[0]):
            # temp_dict[f'{ts_code.iloc[j]}_{date.iloc[j]}'] = feature[j]
            temp_dict[f'{ts_code[j]}_{date[j]}'] = feature[j]
        raw_data_all_in_one[f"pre_{i}"] = temp_dict

    ensure_dir("../dataset/processed_data/experiment/processed_data")
    pickle_dump(f"../dataset/processed_data/experiment/processed_data/raw_data_dict_{label_type}_horizon_{horizon}.pkl", raw_data_all_in_one)


if __name__ == '__main__':
    for l in ["label_7", "label_8"]:
        h = 5
        generator_by_diff_time(l, h)