from tqdm import tqdm
import pandas as pd
import __init__
import recs
from recs.utils import predict_ranking
from recs.models.mf import MatrixFactorization

data = "DATA放這裡"
train_set = recs.datasets.Dataset.from_uir(data, seed=41)

user, item, rate = [], [], []

for i in data:
    user.append(i[0])
    item.append(i[1])
    rate.append(i[2])
    

df = pd.DataFrame({
    "user": user,
    "item": item,
    "rate": rate,
})
df.drop_duplicates(inplace=True)


MF = MatrixFactorization(df, svds_k = 15)

for u in tqdm(set(user)):
    MF.rec_items(u)[MF.pivot_columns].tolist()
