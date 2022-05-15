from sklearn.manifold import TSNE
from sklearn.cluster import Birch, DBSCAN
import numpy as np
import umap

n_cluster = 50
n_comp = 10
mx = 0
mx_ncl = 0
mx_nco = 0

def feature_selecter(udata, method, n_components, random_state=47, n_jobs=30):
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

def clustering(user_data, x_embedded, method, n_clusters=None, eps=None, min_samples=None):
    """
    quick annotation
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
    clusters_top10 = {}
    clusters = user_data.cluster.value_counts().index.tolist()
    for i in clusters:                
        cluster_top10 = tran_data[tran_data['id_number'].isin( 
            user_data[user_data['cluster'] == i]['身分證字號']
        )]['fund_id'].value_counts().index.tolist()[:topn]
        clusters_top10[i] = cluster_top10
    
    return clusters_top10