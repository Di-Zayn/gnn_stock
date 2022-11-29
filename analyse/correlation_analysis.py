import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
import numpy as np
import statsmodels.api as sm


# class CCA:
#     def __init__(self, dataset):
#         self.dataset = dataset


def cca(x, y):
    x, y =x.copy(), y.copy()
    # x, y = np.array(x).reshape(1, -1),  np.array(y).reshape(1, -1)
    x = pd.DataFrame(StandardScaler().fit(x).transform(x), columns=x.columns)
    y = pd.DataFrame(StandardScaler().fit(y).transform(y), columns=y.columns)
    # x = (x - mean) / std
    # y = (y - mean) / std
    n_c = 4
    ca = CCA(n_components=n_c)
    xc, yc = ca.fit(x, y).transform(x, y)
    corr = []
    corr_sum = 0
    for c in range(n_c):
        corr.append(np.corrcoef(xc[:, c], yc[:, c])[0, 1])
        corr_sum += np.corrcoef(xc[:, c], yc[:, c])[0, 1]
    return corr_sum / n_c

def get_param():
    df = pd.read_csv("../dataset/index_csv/daily_data_label_2.csv")
    df = df.drop(columns=["ts_code", "pre_close", "label", "trade_date"])
    cols = df.columns
    mu = []
    sigma = []
    for c in cols:
        mu.append([c, df[c].mean()])
        sigma.append([c, df[c].std(ddof=0)])
    df1 = pd.DataFrame(mu, columns=["index", "mean"])
    df2 = pd.DataFrame(sigma, columns=["index", "std"])
    df1.to_csv("mean.csv", index=False)
    df2.to_csv("std.csv", index=False)


def multi_corr(multi: pd.DataFrame, single: pd.DataFrame):
    x = pd.DataFrame(StandardScaler().fit(multi).transform(multi), columns=multi.columns)
    y = pd.DataFrame(StandardScaler().fit(single).transform(single), columns=single.columns)
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    result = model.fit()
    print(result.summary())
    with open("multi-label2.txt", 'a') as f:
        f.write(str(result.summary()))

if __name__ == "__main__":
    # means = pd.read_csv("mean.csv")["mean"].values
    # stds = pd.read_csv("std.csv")["std"].values
    # means = np.array(means).reshape(1, -1)
    # stds = np.array(stds).reshape(1, -1)

    exp = "label_2"
    pre = []
    for i in range(5):
        pre.append(pd.read_csv(f"./pre_{i}_{exp}.csv").iloc[:, 2:])
    corr_min = np.zeros((5, 5))
    for i in range(5):
        for j in range(i):
            corr_min[i, j] = cca(pre[i], pre[j])
            print(f"pre_{i} & pre_{j}:")
            print(corr_min[i, j])
            with open("rst-avg.txt", 'a') as f:
                f.write(f"pre_{i} & pre_{j}:\n{corr_min[i, j]}\n")
    # for idx in range(len(pre[0])):
    #     if idx > 1:
    #         break
    #     print(f"{idx}:\n")
    #     for i in range(5):
    #         for j in range(i):
    #             corr_min[i, j] = cca(pre[i].loc[idx, :].values, pre[j].loc[idx, :].values, means, stds)
    #             print(f"pre_{i} & pre_{j}:")
    #             print(corr_min[i, j])
    #             # with open("rst-avg.txt", 'a') as f:
    #             #     f.write(f"pre_{i} & pre_{j}:\n{corr_min[i, j]}\n")

    with open("./rst-avg.txt", 'r') as f:
        count = 0
        rst = 0
        while True:
            count += 1
            l = f.readline()
            l.strip()
            if count % 2 == 0:
                rst += float(l)
            if count == 20:
                break
    print("avg", rst/10)