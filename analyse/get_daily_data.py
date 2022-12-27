import talib as tl
import pandas as pd

def cal_custom_index(df, horizon=5, method="max"):
    # 计算 macd 数据
    df['dif'], df['dea'], hist = tl.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = hist * 2
    # 计算均线数据
    # df["MA5"] = tl.MA(df["close"], timeperiod=5)
    df["MA10"] = tl.MA(df["close"], timeperiod=10)
    df["MA20"] = tl.MA(df["close"], timeperiod=20)

    # boll rsi cci
    df["boll_upper"], df["boll_mid"], df["boll_lower"] = tl.BBANDS(df["close"],
                                                                   timeperiod=5, nbdevup=2,
                                                                   nbdevdn=2, matype=0)
    df["RSI"] = tl.RSI(df["close"], timeperiod=14)
    df["CCI"] = tl.CCI(df["high"], df["low"], df["close"], timeperiod=14)

    # KDJ 值对应的函数是 STOCH
    df['k'], df['d'] = tl.STOCH(
        df['high'].values,
        df['low'].values,
        df['close'].values,
        fastk_period=9,
        slowk_period=5,
        slowk_matype=1,
        slowd_period=5,
        slowd_matype=1)
    df['j'] = list(map(lambda x, y: 3 * x - 2 * y, df['k'], df['d']))

    # horizon 前horizon天（包括当天）的close数据
    # method 历史数据的处理方式
    history_close = list()
    for i in range(horizon):
        hst = df['close'].shift(i)
        hst.name = f"close{-i}"
        history_close.append(hst)
    history_close = pd.concat(history_close, axis=1, ignore_index=True)
    # print(history_close.head())
    if method == "max":
        for i in range(len(history_close)):
            history_close.loc[i, f"{method}_close"] = history_close.loc[i, :].max()
    elif method == "mean":
        for i in range(len(history_close)):
            history_close.loc[i, f"{method}_close"] = history_close.loc[i, :].mean()
    elif method == "min":
        for i in range(len(history_close)):
            history_close.loc[i, f"{method}_close"] = history_close.loc[i, :].min()

    df['label'] = ((df['close'].shift(-1) - history_close[f"{method}_close"]) / history_close[f"{method}_close"]) * 100

    # 由于j值的计算需要前33天的数据，所以从34天开始
    # 由于要获得未来一天的close作为标签，所以最后一天的值也是nan
    df = df.dropna()
    # print(df.head())
    return df

def add_label(df, label):
    if label == 1:
        horizon = 1
        method = "max"
    elif label == 2:
        horizon = 3
        method = "max"
    elif label == 3:
        horizon = 5
        method = "max"
    elif label == 4:
        horizon = 7
        method = "max"
    elif label == 5:
        horizon = 5
        method = "mean"
    elif label == 6:
        horizon = 5
        method = "min"
    elif label == 7:
        horizon = 3
        method = "min"
    elif label == 8:
        horizon = 7
        method = "min"

    history_close = list()
    for i in range(horizon):
        hst = df['close'].shift(i)
        hst.name = f"close{-i}"
        history_close.append(hst)
    history_close = pd.concat(history_close, axis=1, ignore_index=True)
    if method == "max":
        for i in range(len(history_close)):
            history_close.loc[i, f"{method}_close"] = history_close.loc[i, :].max()
    elif method == "mean":
        for i in range(len(history_close)):
            history_close.loc[i, f"{method}_close"] = history_close.loc[i, :].mean()
    elif method == "min":
        for i in range(len(history_close)):
            history_close.loc[i, f"{method}_close"] = history_close.loc[i, :].min()

    df[f'label_{label}'] = ((df['close'].shift(-1) - history_close[f"{method}_close"]) / history_close[f"{method}_close"]) * 100
    df = df.dropna()
    return df
def get_daily_data(label):
    data = pd.read_csv("../dataset/history_data.csv")
    data = data.groupby("ts_code")
    custom_index = list()
    if label == "label_1":
        horizon = 1
        method = "max"
    elif label == "label_2":
        horizon = 3
        method = "max"
    elif label == "label_3":
        horizon = 5
        method = "max"
    elif label == "label_4":
        horizon = 7
        method = "max"
    elif label == "label_5":
        horizon = 5
        method = "mean"
    elif label == "label_6":
        horizon = 5
        method = "min"
    for g in data.groups:
        print(f"正在处理:{g}")
        df = data.get_group(g)
        df = df.sort_values(by="trade_date", ignore_index=True)
        df = cal_custom_index(df, horizon, method)
        custom_index.append(df)
    custom_data = pd.concat(custom_index, ignore_index=True)
    custom_data = custom_data[custom_data['trade_date'] >= 20210101]
    custom_data = custom_data[custom_data['trade_date'] < 20220101]
    custom_data.to_csv(f"../dataset/experiment/experiment_data.csv", index=False)

if __name__ == "__main__":
    data = pd.read_csv("../dataset/history_data.csv")
    data = data[data['trade_date'] >= 20201201]
    data = data[data['trade_date'] < 20220101]
    data = data.groupby("ts_code")
    custom_index = list()
    for g in data.groups:
        print(f"正在处理:{g}")
        df_group = data.get_group(g)
        df_group = df_group.sort_values(by="trade_date", ignore_index=True)
        for i in range(7, 9):
            df_group = add_label(df_group, i)
        custom_index.append(df_group)
    custom_data = pd.concat(custom_index, ignore_index=True)
    custom_data.to_csv(f"../dataset/processed_data/experiment/experiment_data_label7-8.csv", index=False)