from sklearn.manifold import TSNE
from sklearn.cluster import Birch, DBSCAN
from recs.utils.common import readjson2dict, id2cat, convert_result_json_2_csv
from recs.utils.path import DATA_PATH, RESULT_PATH
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import umap


def feature_selecter(udata, method,
                     n_components, random_state=47,
                     n_jobs=30):
    """透過 tsne 或 umap 進行特徵提取與降維"""
    if method not in ["tsne", "umap"]:
        raise ValueError("feature select method not allowed")

    if method == "tsne":
        x_embedded = TSNE(
            n_components=n_components,
            init='random',
            random_state=random_state,
            n_jobs=n_jobs
        ).fit_transform(udata.values)
        x_embedded = np.array(x_embedded)

    if method == "umap":
        x_embedded = umap.UMAP(
            n_components=n_components,
            random_state=random_state
        ).fit_transform(udata.values)

    return x_embedded


def clustering(user_data, x_embedded,
               method, n_clusters=None,
               eps=None, min_samples=None):
    """選擇使用 birch 與 dbscan 進行分群
    n_clusters for birch
    eps, min_samples for dbscan
    """
    if method not in ["birch", "dbscan"]:
        raise ValueError("clustering method not allowed")

    if method == "birch":
        brc = Birch(n_clusters=n_clusters)
        brc.fit(x_embedded)
        brc_x = np.array(brc.predict(x_embedded))
        user_data['cluster'] = brc_x

    if method == "dbscan":
        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples
        ).fit(x_embedded)
        user_data['cluster'] = dbscan.labels_

    return user_data


def recommender_method(tran_data, user_data, topn=10):
    """透過 clustering 分群過的 user_data 中的 cluster 欄位label資料，去搜尋該群組的歷史 top10 交易商品
    """
    clusters_topn = {}
    clusters = user_data.cluster.value_counts().index.tolist()
    for i in clusters:
        cluster_top10 = tran_data[tran_data['id_number'].isin(
            user_data[user_data['cluster'] == i]['id_number']
        )]['fund_id'].value_counts().index.tolist()[:topn]
        clusters_topn[i] = cluster_top10

    # 確保個別分群的推薦 item 必須滿足 topn，不滿足的部分以 best_sells 填滿
    for i in cluster_top10:
        cluster_rec_len = len(cluster_top10[i])
        if cluster_rec_len != topn:
            ignore = best_sells['fund_id'].isin(cluster_top10[i])
            new_rec = best_sells[~ignore] \
                .head(topn - cluster_rec_len)['fund_id'] \
                .tolist()
            cluster_top10[i] = cluster_top10[i] + new_rec

    return clusters_topn


if __name__ == "__main__":
    u2idx = readjson2dict("crm_idx")
    i2idx = readjson2dict("fund_info_idx")
    tran_data = pd.read_pickle(DATA_PATH / "trans_buy.pkl")

    # 把 index 化的 id 轉回來
    tran_data["id_number"] = tran_data["id_number"].progress_apply(
        lambda x: id2cat(u2idx, x))
    tran_data["fund_id"] = tran_data["fund_id"].progress_apply(
        lambda x: id2cat(i2idx, x))

    # 讀取分群用的 crm 資料表(透過 recs.datasets.preprocess.crm_new 產生)
    crm = pd.read_csv(DATA_PATH / "crm_for_clustering_2020.csv")
    crm = crm[crm.yyyymm == 202012]

    tran_data = pd.read_pickle(DATA_PATH / "trans_buy.pkl")

    target_users = crm['id_number'].unique().tolist()
    best_sells = tran_data[
        (tran_data.buy_date > 20181200)
    ].fund_id.value_counts().rename_axis('fund_id').reset_index(name='count')

    new_user = crm[crm['has_traded'] == 0]['id_number'].unique().tolist()
    crm[crm['id_number'].isin(new_user)]
    crm.drop(columns=['index'], inplace=True)

    embedding_method = "umap"
    n_components = 25

    user_data = crm.copy()
    u_data = user_data.drop(columns=['id_number', 'CIFAOCODE'])
    u_data = u_data.replace(-np.inf, -1)
    u_data.fillna(0, inplace=True)

    x_embedded = feature_selecter(u_data, embedding_method, n_components)

    c_user_data = clustering(user_data, x_embedded,
                             'dbscan', eps=6.5, min_samples=5)
    cluster_top10 = recommender_method(tran_data, c_user_data)

    result_by_cluster = {}
    for u in tqdm(new_user):
        temp = user_data[user_data['身分證字號'] == u]
        t_cluster = temp.cluster.tolist()[0]
        result_by_cluster[u] = cluster_top10[t_cluster]

    r_name = "r_2022_05_01_cluster"
    json_name = r_name+".json"
    json_save_path = RESULT_PATH / json_name

    with open(json_save_path, 'w') as jf:
        json.dump(result_by_cluster, jf)

    csv_name = f"{r_name}.csv"
    save_path = RESULT_PATH / csv_name
    convert_result_json_2_csv(str(json_save_path), str(save_path))
