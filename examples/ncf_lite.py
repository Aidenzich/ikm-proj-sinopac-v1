import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

from cornac.datasets import amazon_clothing
from cornac.data import Reader

import pandas as pd
import recs


from recs.utils import predict_ranking

data = amazon_clothing.load_feedback(reader=Reader(bin_threshold=1.0))
train_set = recs.datasets.Dataset.from_uir(data, seed=41)


# Instantiate the recommender models to be compared
gmf = recs.models.GMF(
    num_factors=8,
    num_epochs=10,
    learner="adam",
    batch_size=256,
    lr=0.001,
    num_neg=50,
    seed=123,
)
mlp = recs.models.MLP(
    layers=[64, 32, 16, 8],
    act_fn="tanh",
    learner="adam",
    num_epochs=10,
    batch_size=256,
    lr=0.001,
    num_neg=50,
    seed=123,
)
neumf1 = recs.models.NeuMF(
    num_factors=8,
    layers=[64, 32, 16, 8],
    act_fn="tanh",
    learner="adam",
    num_epochs=10,
    batch_size=256,
    lr=0.001,
    num_neg=50,
    seed=123,
)
neumf2 = recs.models.NeuMF(
    name="NeuMF_pretrained",
    learner="adam",
    num_epochs=10,
    batch_size=256,
    lr=0.001,
    num_neg=50,
    seed=123,
    num_factors=gmf.num_factors,
    layers=mlp.layers,
    act_fn=mlp.act_fn,
).pretrain(gmf, mlp)



df = pd.DataFrame(columns=['userId', 'itemId', 'rank'], data=data)

gmf.fit(train_set)
mlp.fit(train_set)
neumf1.fit(train_set)
neumf2.fit(train_set)

pred1 = predict_ranking(gmf, df, 'userId', 'itemId', 'rank', True)
# pred2 = predict_ranking(mlp, df, 'userId', 'itemId', 'rank', True)
# pred3 = predict_ranking(neumf1, df, 'userId', 'itemId', 'rank', True)
# pred4 = predict_ranking(neumf2, df, 'userId', 'itemId', 'rank', True)



print(pred1)
print("========================")
# print(pred2)
# print("========================")
# print(pred3)
# print("========================")
# print(pred4)