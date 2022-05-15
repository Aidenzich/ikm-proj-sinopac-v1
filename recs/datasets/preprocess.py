
from sklearn import preprocessing
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing


from . import *
from ..utils.path import *
from ..utils.common import id2cat, get_cat2id
from ..utils.common import readjson2dict
from ..utils.config import *


def trans_buy():
    """原始資料 trans_buy.csv轉trans_buy.pkl"""
    trans_buy_path = DATA_PATH / "origin/trans_buy.csv"    
    save_path = DATA_PATH / "trans_buy.pkl"
    
    trans_buy = pd.read_csv(trans_buy_path, header=None)
    trans_buy.columns = [
        USER_ID, 'certificate', ITEM_ID, 
        'buy_date','deduction_num', 'deduction_local_amount'
    ]
    
    u2idx = readjson2dict("crm_idx")
    i2idx = readjson2dict("fund_info_idx")
    
    trans_buy = trans_buy[trans_buy[ITEM_ID].isin(i2idx.keys())]
    trans_buy = trans_buy[trans_buy[USER_ID].isin(u2idx.keys())]
    
    trans_buy = trans_buy[[USER_ID, ITEM_ID, 'buy_date']]
    trans_buy['yyyymm'] = trans_buy.buy_date.astype(str).str[:-2].astype(int)
    date = pd.to_datetime(trans_buy['yyyymm'].astype(str), format="%Y%m")
    date = date - pd.DateOffset(months=1)
    trans_buy['yyyymm'] =  date.dt.strftime('%Y%m').astype(int)
    trans_buy.to_pickle(save_path)
    

def crm(cal_var: bool=True, predict_year: int=202012, normalize: bool=True, group_age: bool=True ):
    # crm.csv
    crm_path = DATA_PATH / 'origin/crm.csv'
    save_path = DATA_PATH / 'crm.pkl'
    
    crm = pd.read_csv(crm_path, encoding="Big5", header=None)
    crm.info(memory_usage='deep')
    crm.columns = [
        'yyyymm', 'id_number', 'local_foreign_total', 'local_total', 'local_demand_deposit',     
        'local_fixed_deposit', 'foreign_total', 'foreign_fixed_deposit',
        'foreign_demand_deposit', 'invest_type', 'age', 'monthly_trade_vol',
        'stock_inventory_val', 'KPI'
    ]

    # The target user exist in predict year
    predict_users = crm[crm.yyyymm == predict_year][USER_ID].unique()
    
    crm = crm[crm.id_number.isin(predict_users)]
    u2idx, u_cat = get_cat2id(crm.id_number)
    crm.id_number = u_cat
    crm.fillna(0, inplace=True)
    
    if normalize:
        # norm
        norm_cols = [
            'local_foreign_total', 'local_total', 'local_demand_deposit', 
            'local_fixed_deposit', 'foreign_total',
            'foreign_fixed_deposit', 'foreign_demand_deposit', 
            'monthly_trade_vol', 'stock_inventory_val'
        ]


        for i in norm_cols:
            crm[i] = np.log(crm[i].replace(0,1))
            
        crm.fillna(0, inplace=True)
    
    
    if cal_var:
        save_path = DATA_PATH / 'crm_diff.pkl'
        diff_cols = [
            'local_foreign_total', 'local_total', 'local_demand_deposit', 
            'local_fixed_deposit', 'foreign_total',
            'foreign_fixed_deposit', 'foreign_demand_deposit', 
            'monthly_trade_vol', 'stock_inventory_val'
        ]
        rename_dict = { diff_cols[i]: v+ "_diff" for i, v in enumerate(diff_cols) }

        diff_df = pd.DataFrame([], columns=list(rename_dict.values()))
        for u in tqdm(crm.id_number.unique().tolist()):
            user_df = crm[crm.id_number == u].sort_values(by=["yyyymm"])
            tmp = user_df[diff_cols].diff().replace(np.nan, 0).rename(rename_dict, axis=1)
            diff_df = diff_df.append(tmp)
            
        crm = pd.concat([crm, diff_df], axis=1)
        
    if group_age:
        age_groups = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        crm['age_group'] = pd.cut(crm.age, age_groups).astype(str)
        lb = preprocessing.LabelEncoder()
        crm['age_group'] = lb.fit_transform(crm.age_group)
        
    crm.invest_type = crm.invest_type.astype(int)
    
    # save index 
    idx_save_path = DATA_PATH / "crm_idx.json"
    
    with open(idx_save_path, 'w') as f:
        json.dump(u2idx, f)
    crm.to_pickle(save_path)

def equity():
    # load path
    equity_path = DATA_PATH / 'origin/fund_equity.csv'
    fund_info_path = DATA_PATH / 'fund_info.pkl'
    
    equity = pd.read_csv(equity_path, header=None)
    fund_info = pd.read_pickle(fund_info_path)

    i2idx = readjson2dict("fund_info_idx")
    
    equity.columns = [
        EQUITY_DATE,
        ITEM_ID,
        EQUITY_PRICE
    ]
    
    equity['yyyymmdd'] = equity[EQUITY_DATE].str[:10]
    equity['yyyymmdd'] = equity['yyyymmdd'].str.replace('-', '').astype(int)
    equity['yyyymm'] = equity['yyyymmdd'] // 100
    
    equity.sort_values(by=[ITEM_ID, 'yyyymmdd'], inplace=True)
    equity.reset_index(inplace=True, drop=True)
    
    equity = equity[equity[ITEM_ID].isin(i2idx.keys())]
    equity = equity.drop_duplicates(subset=[ITEM_ID, 'yyyymm'], keep='first')
    
    diff_df = pd.DataFrame([], columns=[
        'diff_month', 'diff_quarter', 'diff_half_year', 'diff_year', 'diff_2year'
    ])

    for i in tqdm(equity.fund_id.unique().tolist()):
        item_df = equity[equity.fund_id == i]
        tmp = item_df[[EQUITY_PRICE]].diff().replace(np.nan, 0).rename({EQUITY_PRICE:'diff_month'}, axis=1)
        tmp['diff_quarter'] = item_df[[EQUITY_PRICE]].diff(periods=3).replace(np.nan, 0).rename({EQUITY_PRICE:'value_diff'}, axis=1)
        tmp['diff_half_year'] = item_df[[EQUITY_PRICE]].diff(periods=6).replace(np.nan, 0).rename({EQUITY_PRICE:'value_diff'}, axis=1)
        tmp['diff_year'] = item_df[[EQUITY_PRICE]].diff(periods=12).replace(np.nan, 0).rename({EQUITY_PRICE:'value_diff'}, axis=1)
        tmp['diff_2year'] = item_df[[EQUITY_PRICE]].diff(periods=24).replace(np.nan, 0).rename({EQUITY_PRICE:'value_diff'}, axis=1)    
        diff_df = diff_df.append(tmp)
        
    equity = pd.concat([equity, diff_df], axis=1)

    equity.drop(columns=[EQUITY_DATE, 'yyyymmdd'], inplace=True)
    equity[ITEM_ID] = equity[ITEM_ID].apply(lambda x: i2idx[x])
    equity = equity.merge(fund_info, how='left', on=[ITEM_ID])
    
    equity_save_path = DATA_PATH / 'equity.pkl'
    equity.to_pickle(equity_save_path)
    
def fund_web():
    fund_info_web_path = DATA_PATH / 'fund_info_web.csv'
    save_path = DATA_PATH / 'fund_info_web.pkl'
    
    fund_info_web = pd.read_csv(fund_info_web_path)
    i2idx, i_cat = get_cat2id(fund_info_web[ITEM_ID])
    
    fund_info_web.drop(columns=['Unnamed: 0'], inplace=True)
    fund_info_web.columns = [
        'fund_id', 'fund_type_web', 'invest_targets',
        'invest_region', 'currency', 'scale', 'manager'
    ]
    
    fund_info_web.fillna('0', inplace=True)
    
    lb_cols = ['fund_type_web', 'invest_targets', 'invest_region', 'currency' ]
    lb = preprocessing.LabelEncoder()
    for i in lb_cols:
        fund_info_web[i] = lb.fit_transform(fund_info_web[i])
        
    fund_info_web.drop(columns=['scale', 'manager'], inplace=True)
    fund_info_web.to_pickle(save_path)
    
    
def fund():
    # load path
    fund_info_path = DATA_PATH / 'origin/fund_info.csv'
    fund_info_web_path = DATA_PATH / 'fund_info_web.pkl'
    
    # save path
    idx_save_path = DATA_PATH / 'fund_info_idx.json'
    save_path = DATA_PATH / 'fund_info.pkl'
    
    fund_info = pd.read_csv(fund_info_path, encoding='Big5', header=None)
    fund_info.columns = [
        ITEM_ID, 'chinese_name', 'region',
        'fund_type', 'AUM', 'local_or_foreign',
        'currency_type','chosen','guaranteed'
    ]
    
    fund_info_web = pd.read_pickle(fund_info_web_path)
    funds_on_web = fund_info_web.fund_id.tolist()
    
    fund_info.drop(columns=['chinese_name'], inplace=True)

    fund_info = fund_info[fund_info.fund_id.isin(funds_on_web)]
    fund_info = fund_info.merge(fund_info_web, how='left', on=[ITEM_ID])
    
    i2idx, i_cat = get_cat2id(fund_info.fund_id)

    fund_info.fund_id = i_cat
    fund_info['fund_type'] = fund_info['fund_type'].astype(str)

    lb_cols = ['fund_type', 'guaranteed']
    lb = preprocessing.LabelEncoder()
    
    for col in lb_cols:
        fund_info[col] = lb.fit_transform(fund_info[col])

    
    onehot_cols = ['AUM', 'region', 'currency']
    fund_info = pd.get_dummies(fund_info, columns=onehot_cols)
    fund_info.to_pickle(save_path)
    
    with open(idx_save_path, 'w') as f:
        json.dump(i2idx, f)
        
def crm_new(savename="crm_for_clustering_2020.csv"):
    crm_new = pd.read_csv("crm_new.csv", encoding="big5")
    crm_non_trade = pd.read_csv("fund_crm_feature.csv", encoding="big5")
    crm_non_trade.columns = crm_non_trade.columns.str.replace('YYYYMM', 'yyyymm')    
    
    # 僅保留 crm_new 與 fund_crm_featrue 資料表中的共有欄位
    crm_intersec_col = list(set(crm_new.columns) & set(crm_non_trade.columns)) 
    crm_new = crm_new[crm_intersec_col]
    crm_new['has_traded'] = 1
    
    crm_non_trade = crm_non_trade[crm_intersec_col]
    crm_non_trade['has_traded'] = 0
    
    crm_for_clustering = pd.concat([crm_new, crm_non_trade]).reset_index()
    crm_non_trade.columns = crm_non_trade.columns.str.replace('身分證字號', 'id_number')
    
    
    # 此處欄位請隨輸入資料調整
    norm_numerical_col = []
    numerical_col = []
    multi_category_col = []
    special_col = []
    unique_col = []
    useless_col = [] # 無用處的欄位，值都一樣
    
    crm_for_clustering.drop(useless_col, axis=1, inplace=True)
    crm_for_clustering.fillna(0, inplace=True)
    
    for i in norm_numerical_col:
        if i in crm_for_clustering.columns:
            crm_for_clustering[i] = np.log(crm_for_clustering[i].replace(0,1))
            
    crm_for_clustering = pd.get_dummies(crm_for_clustering, columns=multi_category_col)
    crm_for_clustering = crm_for_clustering.replace(-np.inf, -1)
    
    crm_for_clustering.to_csv(savename, index=False)
    
    
    