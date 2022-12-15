# 用于统计正负样本的比例
from collections import Counter

from dataset.nn_data_generator import BaseSet

feature_source = "v2_feature"
item = ['train', 'val']
mode = "label_2_horizon_5_2021-2022_5"
set_path = f"./dataset/processed_data/{feature_source}"
for i in item:
    dataset = BaseSet(filepath=set_path, filename=f"corr_processed_{i}_data_index_{mode}.pkl", label_log=True)
    label = dataset.load_data()
    print(f'Data Load Success data:{len(dataset.label)}')
    print("Rate:", Counter(label))