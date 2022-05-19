# %%
import sys

import pandas as pd
from recs.utils.path import *
from recs.utils.common import readjson2dict, id2cat
from lightgbm import LGBMRegressor
from tqdm import tqdm
import shap
import numpy as np
tqdm.pandas()


#%%
u2idx=readjson2dict("crm_idx")
i2idx=readjson2dict("fund_info_idx")

pred = pd.read_csv(DATA_PATH / "pred.csv")
pred['userId'] = pred['userId'].progress_apply(lambda x: id2cat(u2idx, x))
pred['itemId'] = pred['itemId'].progress_apply(lambda x: id2cat(i2idx, x))
pred

crm = pd.read_csv(DATA_PATH / "crm_for_clustering_2020.csv")
crm['userId'] = crm['id_number'] 
crm.drop(columns=['id_number', 'index'], inplace=True)

fund = pd.read_pickle(DATA_PATH / 'fund_info.pkl')
fund['itemId'] = fund['fund_id'].progress_apply(lambda x: id2cat(i2idx, x))
fund.drop(columns=['fund_id', 'currency_type'], inplace=True)

pred = pred.merge(crm, how='left', on=['userId'])
pred = pred.merge(fund, how='left', on=['itemId'])

keys = pred[['userId', 'itemId']]
y = pred['rating']
x = pred.drop(columns=['userId', 'itemId', 'rating', 'CIFAOCODE'])

model = LGBMRegressor(
    boosting_type="gbdt", 
    verbose=1, 
    random_state=47
)
model.fit(x, y)
model.predict([x.values[0]])

explainer = shap.Explainer(model)
shap_vals = explainer(x)
feature_names = shap_vals.feature_names
print(len(feature_names))

value_df_dict = {}
base_val = shap_vals.base_values[0]

for i in tqdm(shap_vals):
    i_sum = i.values.sum()
    for jidx, j in enumerate(feature_names):        
        data_val = i.data[jidx]
        val = round(i.values[jidx] / (i_sum - base_val), 4)
        save_val = (val, data_val)
        if value_df_dict.get(j, 0) == 0:
            value_df_dict[j] = [save_val]
        else:
            value_df_dict[j].append(save_val)
    

df = pd.DataFrame(value_df_dict)


# %%
df['top5'] = df.apply(
    lambda x:list(df.columns[np.array(x[::1]).argsort()[::-1][:5]]), 
    axis=1
)
df['user'] = keys['userId']
df['item'] = keys['itemId']


df['top5'] = df['top5'].progress_apply(lambda x: ",".join(x))
cols = df.columns.tolist()
cols = cols[-2:] + cols[:-2]
df = df[cols]

df.to_csv('explain.csv')