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


if __name__ == "__main__":
    label = "label_2"
    hist_horizon = 5

    daily_data = pd.read_csv(f"../dataset/history_data.csv")

    # 被预测的数据
    nxt = daily_data[["ts_code", "trade_date", "label"]].copy()
    nxt = nxt.groupby("ts_code")
    nxt_list = []
    for g in nxt.groups:
        mini_nxt = nxt.get_group(g)[hist_horizon - 1:]
        nxt_list.append(mini_nxt)
    nxt = pd.concat(nxt_list, ignore_index=True)
    nxt.columns=["ts_code", "trade_date", "label"]
    nxt.to_csv(f"./nxt_{label}_horizon_{hist_horizon}.csv", index=False)

    # 将历史数据提取出来 并剔除部分列
    if 'pre_close' in daily_data.columns:
        daily_data = daily_data.drop(columns=["pre_close", "label"])
    else:
        daily_data = daily_data.drop(columns=["label"])
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
        data.to_csv(f"./pre_{i}_{label}_horizon_{hist_horizon}.csv", index=False)