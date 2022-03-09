import os
import pandas as pd
import numpy as np
import recs.path as path
from tqdm import tqdm

fund_info_path = os.path.join(path.data_path, 'fund_info.pkl')
crm_path = os.path.join(path.data_path, 'crm.pkl')
fund= pd.read_pickle(fund_info_path)
crm=  pd.read_pickle(crm_path)

print(fund.info(memory_usage='deep'))
print(crm.info(memory_usage='deep'))

fund

yyyymmlist = crm.yyyymm.unique().tolist()

fund['k'] = 0
crm['k'] = 0
print('Merging CRM and Fund Datas...')
count = 1
# for batch in tqdm(np.array_split(crm, 600)):
for batch in tqdm(yyyymmlist):
    tmp = crm[crm.yyyymm == batch]
    tmp = tmp.merge(fund, how='outer')
#     tmp = batch.merge(fund, how='outer')
    save_name = f"{batch}_m.pkl"
    save_path = os.path.join(path.data_path, 'merge')
    save_path = os.path.join(save_path, save_name)
    tmp.to_pickle(save_path)
