import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score


def cluster(src):
    k = get_best_k(src)
    estimator = MiniBatchKMeans(n_clusters=k, random_state=1, batch_size=2048)
    # estimator.fit(img)
    result = estimator.cluster_centers_

def get_best_k(src):
    k_list = [i for i in range(7, 9)]
    scores = []
    for i in k_list:
        estimator = MiniBatchKMeans(n_clusters=i, random_state=1, batch_size=2048)
        scores.append(silhouette_score(src, estimator.fit_predict(src), sample_size=int(src.shape[0] / 128)))
    index = scores.index(max(scores))
    best_k = k_list[index]
    print(f'best_k:', best_k)
    return best_k

if __name__ == "__main__":
    ## 第一种是选一个股票 来聚类
    ## 第二种是所有股票都输进去聚类
    df = pd.read_csv("../dataset/history_data.csv")
    df = df[df['trade_date'] < 20200501]
    