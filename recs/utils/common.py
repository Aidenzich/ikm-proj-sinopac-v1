import pandas as pd
import time
from tqdm import tqdm
from recs.utils.path import DATA_PATH, MODEL_PATH, EXPORT_PATH
import json
import joblib
import numpy as np
import numbers


def sigmoid(x):
    """Sigmoid function"""
    return 1.0 / (1.0 + np.exp(-x))


def scale(values, target_min, target_max, source_min=None, source_max=None):
    """Scale the value of a numpy array "values" from source_min,
       source_max into a range [target_min, target_max]
    """
    if source_min is None:
        source_min = np.min(values)
    if source_max is None:
        source_max = np.max(values)
    if source_min == source_max:  # improve this scenario
        source_min = 0.0

    values = (values - source_min) / (source_max - source_min)
    values = values * (target_max - target_min) + target_min
    return values


def get_random_state(seed):
    '''Return a RandomState of Numpy.
    If seed is None, use RandomState singleton from numpy.
    If seed is an integer, create a RandomState from that seed.
    If seed is already a RandomState, just return it.
    '''
    if seed is None:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("Fail to create a numpy.random.RandomState")


def estimate_number_batches(input_size, batch_size):
    """
    Estimate number of batches give `input_size` and `batch_size`
    """
    return int(np.ceil(input_size / batch_size))


def clip(values, lower_bound, upper_bound):
    """Enforce values to lie in the specific range (lower_bound, upper_bound)
    """
    values = np.where(values > upper_bound, upper_bound, values)
    values = np.where(values < lower_bound, lower_bound, values)

    return values


def timer(func):
    def wrapper(*args, **kwargs):
        s = time.perf_counter()
        v = func(*args, **kwargs)
        e = time.perf_counter()
        print(f"{func.__name__} takes {e-s} s.")
        return v
    return wrapper


def save_model(model, filename=None):
    if filename is None:
        filename = str(time.time()) + '_model'

    model_path = MODEL_PATH / filename
    joblib.dump(model, model_path)


def save2json(save_object, filename=None, folder=""):
    if (filename is None):
        filename = str(time.time()) + ".json"

    filepath = EXPORT_PATH / folder / filename

    with open(filepath, 'w') as jf:
        json.dump(save_object, jf)

    return filepath


def readjson2dict(filename):
    filename = filename + ".json"
    filepath = DATA_PATH / filename
    with open(filepath) as jf:
        json_dict = json.load(jf)
    return json_dict


def recall_evaluate(rec_dict, real_dict):
    recall3_sum, recall5_sum, recall10_sum, recall_all_sum = 0, 0, 0, 0
    evaluatelen = len(real_dict)

    for i in tqdm(real_dict):
        real_list = real_dict[i]
        if len(real_list) == 0:
            evaluatelen -= 1
            continue

        rec_list = rec_dict.get([i], [])

        recall3 = len(list(set(rec_list[:3]) & set(real_list))) \
            / len(real_list)
        recall5 = len(list(set(rec_list[:5]) & set(real_list))) \
            / len(real_list)
        recall10 = len(list(set(rec_list[:10]) & set(real_list))) \
            / len(real_list)
        recall_all = len(list(set(rec_list) & set(real_list))) / len(real_list)

        recall3_sum += recall3
        recall5_sum += recall5
        recall10_sum += recall10
        recall_all_sum += recall_all

    result = {
        "recall@3": recall3_sum/evaluatelen,
        "recall@5": recall5_sum/evaluatelen,
        "recall@10": recall10_sum/evaluatelen,
        "recall@all": recall_all_sum/evaluatelen
    }
    return result


def id2cat(cat_dict, idx):
    return list(cat_dict.keys())[list(cat_dict.values()).index(idx)]


def get_cat2id(pandas_series):
    cats = pandas_series.astype('category').cat.codes
    cat_dict = dict(zip(pandas_series, cats))
    return cat_dict, cats


def memory_usage(pandas_obj):
    usage_mb = 0
    if isinstance(pandas_obj, pd.DataFrame):
        usage_byte = pandas_obj.memory_usage(deep=True).sum()
    else:
        usage_byte = pandas_obj.memory_usage(deep=True)
    usage_mb == usage_byte / 1024 ** 2
    return "{:03.2f} MB".format(usage_mb)


def convert_result_json_2_csv(jsonfile: str, savename: str, topn=10):
    with open(jsonfile) as jf:
        result_json = json.load(jf)

    result_json_value = list(result_json.values())

    fundlist = [[] for _ in range(topn)]

    for i in range(topn):
        fundlist[i] = (list(map(lambda x: x[i], result_json_value)))

    df_dict = {}
    df_dict['uid'] = result_json.keys()
    for i in range(topn):
        df_dict[f"fundid{i}"] = fundlist[i]

    df = pd.DataFrame(df_dict)

    if '.csv' not in savename:
        savename += '.csv'
    df.to_csv(savename)
    print(f"Convert {jsonfile} to {savename} success.")


def predict_ranking(
    model, data,
    usercol, itemcol,
    predcol, remove_seen=False,
):
    users, items, preds = [], [], []
    item = list(model.train_set.iid_map.keys())

    for uid, user_idx in model.train_set.uid_map.items():
        user = [uid] * len(item)
        users.extend(user)
        items.extend(item)
        preds.extend(model.score(user_idx).tolist())

    all_predictions = pd.DataFrame(
        data={usercol: users, itemcol: items, predcol: preds}
    )

    if remove_seen:
        tempdf = pd.concat(
            [
                data[[usercol, itemcol]],
                pd.DataFrame(
                    data=np.ones(data.shape[0]), columns=["dummycol"],
                    index=data.index
                ),
            ],
            axis=1,
        )
        merged = pd.merge(tempdf, all_predictions,
                          on=[usercol, itemcol], how="outer")
        return merged[merged["dummycol"].isnull()].drop("dummycol", axis=1)
    else:
        return all_predictions
