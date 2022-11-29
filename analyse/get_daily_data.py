import tushare as ts
import talib as tl
import pandas as pd

def get_ts_code():
    df = pd.read_csv("../dataset/index_csv/index_daily.csv", usecols=["ts_code"])
    df = df.groupby("ts_code")
    code = df.groups
    return list(code)

def get_index_daily(token):
    """
    实际没使用 因为积分不够
    """
    ts.set_token(token)
    pro = ts.pro_api()

    # 获取交易日信息, 得到21年的开始日和结束日
    # 是否应该用这个date？
    trade_cal = pro.trade_cal(exchange='', start_date='20210101', end_date='20211224',
                              fields='exchange, cal_date,is_open, pretrade_date', is_open='1')
    start_date, end_date = trade_cal.loc[0]["cal_date"], trade_cal.loc[len(trade_cal) - 1, "cal_date"]
    print(start_date, end_date)

    # 获取ts_code
    # ts_code = pro.index_basic(market='SSE', fields="ts_code")
    # print(ts_code)
    ts_code = get_ts_code()
    print(ts_code)

    # 获取全年各股的股票信息 for循环得到index_daily,然而权限不够...
    data_list = list()
    for code in ts_code:
        print(code)
        df = pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
        print(df.head())
        data_list.append(df)
    index_daily = pd.concat(data_list, ignore_index=True)
    return index_daily

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

def get_daily_data(label):
    data = pd.read_csv("../dataset/index_csv/index_daily.csv")
    data = data.drop(columns=["label"])
    data = data.groupby("ts_code")
    custom_index = list()
    if label == "label_1":
        horizon = 1
        method = "max"
    elif label == "label_2":
        horizon = 5
        method = "max"
    elif label == "label_3":
        horizon = 3
        method = "max"
    elif label == "label_4":
        horizon = 7
        method = "max"
    elif label == "label_5":
        horizon = 5
        method = "mean"
    for g in data.groups:
        print(f"正在处理:{g}")
        df = data.get_group(g)
        # tushare的数据是降序，第一行是最新数据，所以需要先排序
        # iquant拿到的数据是升序，并且已经添加了自定义特征，所以实际上没走这个函数
        df = df.sort_values(by="trade_date", ignore_index=True)
        df = cal_custom_index(df, horizon, method)
        custom_index.append(df)
    custom_data = pd.concat(custom_index, ignore_index=True)
    custom_data.to_csv(f"../dataset/index_csv/daily_data_{label}.csv", index=False)

if __name__ == "__main__":
    # daily_data.csv里的原label是不同表（hsgt idnex_daily...）聚合方法下的label
    my_token = "24587244498b9b7854973468a660fb928a04da68fbc38819a3b4eb33"
    experiment = "label_3"
    get_daily_data(experiment)