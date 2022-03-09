from site import USER_BASE
from turtle import Turtle
import pandas as pd
from . import *
from ..utils.path import *
from ..models.mf import MatrixFactorization


def load_data(cal_diff):
    fund_info = pd.read_pickle(DATA_PATH / "fund_info.pkl")
    trans_buy = pd.read_pickle(DATA_PATH / "trans_buy.pkl")
    equity = pd.read_pickle(DATA_PATH / "equity.pkl")
    crm = pd.read_pickle(DATA_PATH / "crm_diff.pkl" if cal_diff else "crm.pkl")    
    return fund_info, trans_buy, equity, crm
    
def get_mf_result(
    train_start:int,
    train_end:int,
    best_sell_start:int,
    best_sell_end:int,
    best_sell_top_n: int,
    svds_k: int
    ):
    """_summary_

    Args:
        train_start (int): 訓練用資料起始日
        train_end (int): 訓練用資料結束日
        best_sell_start (int): 熱銷資料起始日
        best_sell_end (int): 熱銷資料結束日
        best_sell_top_n (int): 取交易頻率前n筆
        svds_k (int): svd K
    """
    
    fund_info, trans_buy, equity, crm = load_data(cal_diff=True)
    
    train_df = trans_buy[
        (trans_buy.buy_date < train_end) &
        (trans_buy.buy_date > train_start)
    ]
    
    best_sell_df = trans_buy[
        (trans_buy.buy_date < best_sell_end) &
        (trans_buy.buy_date > best_sell_start)
    ][ITEM_ID].value_counts().rename_axis(ITEM_ID).reset_index(
        name="count"
    )
    
    train_df = train_df.groupby([USER_ID, ITEM_ID]).size().reset_index(
        name="rate"
    )
    train_df = train_df[train_df[ITEM_ID].isin(
        best_sell_df[ITEM_ID].tolist()[:best_sell_top_n]
    )]
    
    MF = MatrixFactorization(
        train_df, 
        svds_k=svds_k,
        pivot_index_name=USER_ID,
        pivot_columns_name=ITEM_ID,
        pivot_values_name='rate'
    )
    
    trans_user = trans_buy[USER_ID].unique().tolist()
    trans_freq = trans_buy[USER_ID].value_counts()
    
    trans_buy['y'] = 1
    
    negatives = {}
    negatives[USER_ID] = []
    negatives[ITEM_ID] = []
    negatives["yyyymm"] = []
    negatives["y"] = []
    
    import tqdm
    for user in tqdm(trans_user):
        buy_funds = trans_buy[trans_buy[USER_ID]][ITEM_ID].tolist()
        buy_yyyymm = trans_buy[trans_buy[USER_ID]]["yyyymm"].tolist()
        buy_dates = trans_buy[trans_buy[USER_ID]].buy_date.tolist()
        
        mf_sample = MF.rec_items(user, topn=best_sell_top_n)