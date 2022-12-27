import pandas as pd

def get_hist(df, horizon, cols):
    """
    用于增加历史指标列（如前一天的kdj指标）
    列名：pre_col_horizon 前horizon天的col指标
    """
    x = df.copy()
    for col in cols:
        x.loc[:, col] = df[col].shift(horizon)
    return x

def extract_hist(hist_horizon):
    daily_data = pd.read_csv(f"../dataset/processed_data/experiment/experiment_data.csv")
    # 将历史数据提取出来 并剔除部分列
    if 'pre_close' in daily_data.columns:
        daily_data = daily_data.drop(columns=["pre_close", "label", "label_1", "label_2", "label_3", "label_4", "label_5", "label_6"])
    else:
        daily_data = daily_data.drop(columns=["label", "label_1", "label_2", "label_3", "label_4", "label_5", "label_6"])
    columns = daily_data.columns.values.tolist()
    columns.remove("ts_code")
    columns.remove("trade_date")
    daily_data = daily_data.groupby("ts_code")

    groups = daily_data.groups
    for i in range(hist_horizon):
        df_list = []
        for g in groups:
            grp_df = daily_data.get_group(g)
            grp_df = get_hist(grp_df, i, columns)
            grp_df = grp_df[hist_horizon - 1:]
            df_list.append(grp_df)
        data = pd.concat(df_list, ignore_index=True)
        data = data.dropna()
        data = data[data['trade_date'] >= 20210101]
        data.to_csv(f"../dataset/processed_data/experiment/pre_{i}.csv", index=False)

def extract_target(label):
    daily_data = pd.read_csv(f"../dataset/processed_data/experiment/experiment_data_label7-8.csv")
    # 被预测的数据
    nxt = daily_data[["ts_code", "trade_date", label]].copy()
    # nxt = nxt.groupby("ts_code")
    # nxt_list = []
    # for g in nxt.groups:
    #     mini_nxt = nxt.get_group(g)[hist_horizon - 1:]
    #     nxt_list.append(mini_nxt)
    # nxt = pd.concat(nxt_list, ignore_index=True)
    nxt.columns = ["ts_code", "trade_date", "label"]
    nxt = nxt[nxt['trade_date'] >= 20210101]
    nxt.to_csv(f"../dataset/processed_data/experiment/nxt_{label}.csv", index=False)

if __name__ == "__main__":
    for l in ["label_7", "label_8"]:
        extract_target(l)
    # extract_hist(7)

