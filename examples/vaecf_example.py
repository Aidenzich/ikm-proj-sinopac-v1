import __init__
import recs
import pandas as pd
from recs.utils.path import DATA_PATH
from recs.utils.common import predict_ranking, readjson2dict



WITH_TIME = True

# Read necessary data
trans_buy = pd.read_pickle(DATA_PATH / "trans_buy.pkl")
crm = pd.read_pickle(DATA_PATH / "crm_diff.pkl")
u2idx = readjson2dict("crm_idx")
i2idx = readjson2dict("fund_info_idx")

if WITH_TIME:
    trans_buy["t_weight"] = trans_buy.buy_date.apply(lambda x: pow(
        (x - trans_buy.buy_date.min() + 1) / (trans_buy.buy_date.max() - trans_buy.buy_date.min()), 2
    ))
    rate_df = trans_buy.groupby(["id_number", "fund_id"]).agg({"t_weight":"sum"}).reset_index().rename(
        columns={"t_weight": "rating"}
    )
else:
    rate_df = trans_buy.groupby(["id_number", "fund_id"]).size().reset_index(name="rating")
    
best_sells = trans_buy[trans_buy.buy_date > 201900].fund_id.value_counts().rename_axis("fund_id").reset_index(name="count")
data = rate_df[["id_number", "fund_id", "rating"]].values


train_set = recs.datasets.Dataset.from_uir(data, seed=41)

vaecf = recs.models.VAECF(
    k=10,
    autoencoder_structure=[20],
    act_fn="tanh",
    likelihood="mult",
    n_epochs=100,
    batch_size=100,
    learning_rate=0.001,
    beta=1.0,
    seed=123,
    use_gpu=True,
    verbose=True,
)

vaecf.fit(train_set)

df = pd.DataFrame(columns=['userId', 'itemId', 'rank'], data=data)
pred = predict_ranking(vaecf, df, 'userId', 'itemId', 'rank', True)

print(pred)