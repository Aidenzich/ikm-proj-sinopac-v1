import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
from cornac.datasets import amazon_clothing
from cornac.data import Reader

import recs
from recs.utils.common import predict_ranking
import pandas as pd

# Load data by cornac, with shape [total_trans_num, 3]
data = amazon_clothing.load_feedback(reader=Reader(bin_threshold=1.0))

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