import os
import pandas as pd
from tqdm import tqdm
from .utils.path import LOG_PATH
from .utils.config import USER_ID, ITEM_ID


def get_backtest_answer(
    trans_df: pd.DataFrame, target_users=[],
    start: int = 20160000, end: int = 20200000
):

    DATE_KEY = "buy_date"

    ans_df = trans_df[
        (trans_df[DATE_KEY] > start) &
        (trans_df[DATE_KEY] <= end)
    ]
    if (len(target_users) == 0):
        ans_users = ans_df[USER_ID].unique().tolist()
    else:
        ans_users = ans_df[ans_df[USER_ID].isin(target_users)][USER_ID] \
            .unique().tolist()

    ans_dict = {}
    for u in tqdm(ans_users):
        ignore_items = trans_df[
            (trans_df[USER_ID] == u) &
            (trans_df[DATE_KEY] < start)
        ][ITEM_ID].unique().tolist()

        ans_dict[u] = ans_df[
            ~(ans_df[ITEM_ID].isin(ignore_items)) &
            (ans_df[USER_ID] == u)            
        ][ITEM_ID].unique().tolist()

    return ans_dict


def model_score_logger(model_name: str, params: dict, scores: dict):
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    log_path = os.path.join(LOG_PATH, 'model_score_log')
    log_str = f"ModelName:[{model_name}] "
    for i in params:
        log_str += f"Params[{i}]:{params[i]} "
    for i in scores:
        log_str += f"Scores[{i}]:{scores[i]} "

    log_str += "\n"

    with open(log_path, 'a') as log:
        log.write(log_str)
