# %%
# import sys
# sys.path.append('./')
# sys.path.append('../')
# sys.path.append('../../')
import recs
import pandas as pd
from recs.utils.common import predict_ranking

data = ...
train_set = recs.datasets.Dataset.from_uir(data, seed=41)
print(train_set.num_items)
print(train_set.num_users)
# %%
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

df = pd.DataFrame(columns=['userId', 'itemId', 'rating'], data=data)
pred = predict_ranking(vaecf, df, 'userId', 'itemId', 'rating', True)

print(pred)
# %%
pred.shape

# %%
pred[pred.userId == '196'].rating.tolist()

# %%
df
# %%
