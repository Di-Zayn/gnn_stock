import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
from collections import Counter
from data_script import retrieve_name

sys.path.append("../")
from utils.file_process import ensure_dir, pickle_dump

def generator_by_diff_class():
    """
    构建数据,依据的图谱是由不同类型的表构成的，包括index_daily, daily_basic等等
    """
    # ts_code, trade_date
    index_daily = pd.read_csv(f"../dataset/index_csv/index_daily.csv", encoding="gbk")
    daily_basic = pd.read_csv(f"../dataset/index_csv/daily_basic.csv", encoding="gbk")
    # trade_date
    margin = pd.read_csv(f"../dataset/index_csv/margin.csv", encoding="gbk")
    hsgt = pd.read_csv(f"../dataset/index_csv/hsgt.csv", encoding="gbk")
    # date
    shibor = pd.read_csv(f"../dataset/index_csv/shibor.csv", encoding="gbk")
    us_tycr = pd.read_csv(f"../dataset/index_csv/us_tycr.csv", encoding="gbk")
    us_tbr = pd.read_csv(f"../dataset/index_csv/us_tbr.csv", encoding="gbk")
    us_tltr = pd.read_csv(f"../dataset/index_csv/us_tltr.csv", encoding="gbk")

    # month
    cn_cpi = pd.read_csv(f"../dataset/index_csv/cn_cpi.csv", encoding="gbk")
    cn_ppi = pd.read_csv(f"../dataset/index_csv/cn_ppi.csv", encoding="gbk")

    ensure_dir("../dataset/processed_data")
    label_dict = dict()
    # copy后for循环会更快？
    llabel = index_daily[['ts_code', 'trade_date', 'label']].copy()
    # print(Counter(index_daily['label']))

    # 将index_daily的label提取出来
    for i in range(llabel.shape[0]):
        label_dict[f"{llabel['ts_code'][i]}_{llabel['trade_date'][i]}"] = llabel['label'][i]
    raw_data_all_in_one = {"label": label_dict}
    index_daily = index_daily.drop(labels="label", axis=1)

    # 将index_daily中其他列作为特征提取出来，并做归一化
    for item in [index_daily, daily_basic]:
        temp_dict = dict()
        fea_data = np.array(item.iloc[:, 2:].copy())
        # 为什么这样归一化？
        fea_data = (fea_data - np.min(fea_data, axis=0) + 1) / (np.max(fea_data, axis=0) - np.min(fea_data, axis=0) + 2)
        ts_code = item['ts_code'].copy()
        date = item['trade_date'].copy()
        for i in range(item.shape[0]):
            temp_dict[f'{ts_code[i]}_{date[i]}'] = fea_data[i]
        raw_data_all_in_one[retrieve_name(item)] = temp_dict

    for item in [hsgt, margin]:
        temp_dict = dict()
        fea_data = np.array(item.iloc[:, 1:].copy())
        fea_data = (fea_data - np.min(fea_data, axis=0) + 1) / (np.max(fea_data, axis=0) - np.min(fea_data, axis=0) + 2)
        date = item['trade_date'].copy()
        for i in range(item.shape[0]):
            temp_dict[f'{date[i]}'] = fea_data[i]
        raw_data_all_in_one[retrieve_name(item)] = temp_dict

    for item in [shibor, us_tycr, us_tbr, us_tltr]:
        temp_dict = dict()
        fea_data = np.array(item.iloc[:, 1:].copy())
        fea_data = (fea_data - np.min(fea_data, axis=0) + 1) / (np.max(fea_data, axis=0) - np.min(fea_data, axis=0) + 2)
        date = item['date'].copy()
        for i in range(item.shape[0]):
            temp_dict[f'{date[i]}'] = fea_data[i]
        raw_data_all_in_one[retrieve_name(item)] = temp_dict

    for item in [cn_cpi, cn_ppi]:
        temp_dict = dict()
        fea_data = np.array(item.iloc[:, 1:].copy())
        fea_data = (fea_data - np.min(fea_data, axis=0) + 1) / (np.max(fea_data, axis=0) - np.min(fea_data, axis=0) + 2)
        date = item['month'].copy()
        for i in range(item.shape[0]):
            temp_dict[f'{date[i]}'] = fea_data[i]
        raw_data_all_in_one[retrieve_name(item)] = temp_dict

    pickle_dump(f"../dataset/processed_data/raw_data_index.pkl", raw_data_all_in_one)


def generator_by_diff_time():
    experiment = "label_2"
    horizon = 5
    pre = []
    for i in range(horizon):
        pre.append(pd.read_csv(f"../analyse/pre_{i}_{experiment}_horizon_{horizon}.csv"))
    ensure_dir("../dataset/processed_data")
    label_dict = dict()

    label = pd.read_csv(f"../analyse/nxt_{experiment}_horizon_{horizon}.csv")
    # print(Counter(label['label']))

    # 将index_daily的label提取出来
    for i in range(label.shape[0]):
        label_dict[f"{label['ts_code'][i]}_{label['trade_date'][i]}"] = label['label'][i]
    raw_data_all_in_one = {"label": label_dict}

    ts_code = label['ts_code'].copy()
    date = label['trade_date'].copy()

    for i in range(horizon):
        temp_dict = {}
        feature = np.array(pre[i].iloc[:, 2:].copy())
        # 为何这样归一化？
        # 为何不是std之类的...
        feature = (feature - np.min(feature, axis=0) + 1) / (np.max(feature, axis=0) - np.min(feature, axis=0) + 2)
        for j in range(pre[i].shape[0]):
            temp_dict[f'{ts_code[j]}_{date[j]}'] = feature[j]
        raw_data_all_in_one[f"pre_{i}"] = temp_dict

    pickle_dump(f"../dataset/processed_data/raw_data_dict_{experiment}_horizon_{horizon}.pkl", raw_data_all_in_one)


if __name__ == '__main__':
    generator_by_diff_time()