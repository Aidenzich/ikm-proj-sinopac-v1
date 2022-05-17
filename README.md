# Sinopac Recsys
## Repository Structure
```
.
├── .gitignore
├── NEW_COLUMNS.MD
├── README.md
├── examples
│   ├── __init__.py
│   ├── mf_example.py
│   ├── ncf_example.py
│   └── vaecf_example.py
├── experience
│   ├── classifier
│   │   ├── ClassifiersResult.ipynb
│   │   ├── ClassifiersTest-Copy1.ipynb
│   │   ├── ClassifiersTest-Copy2.ipynb
│   │   ├── ClassifiersTest-Copy3.ipynb
│   │   ├── ClassifiersTest-Copy4.ipynb
│   │   ├── ClassifiersTest-Copy5.ipynb
│   │   ├── ClassifiersTest-xgboost.ipynb
│   │   ├── ClassifiersTest.ipynb
│   │   └── model_with_diff.ipynb
│   ├── cluster
│   │   ├── Untitled.ipynb
│   │   ├── cluster_pred.py
│   │   ├── exp1.ipynb
│   │   ├── exp2.ipynb
│   │   ├── exp3.ipynb
│   │   ├── exp4.ipynb
│   │   ├── exp5_nontrade.ipynb
│   │   └── explain.ipynb
│   ├── collaborative_filtering
│   │   ├── MFResult.ipynb
│   │   ├── MFTest.ipynb
│   │   ├── do_cf_group.ipynb
│   │   ├── mf_prob.ipynb
│   │   ├── mf_prob_history.ipynb
│   │   └── mf_torch.ipynb
│   ├── mf_example.ipynb
│   └── others
│       ├── SMOTE-Scatter.ipynb
│       ├── best_sell_cover_rate.ipynb
│       ├── merge_by_yyyymm.py
│       ├── result_json2csv.ipynb
│       ├── shap.ipynb
│       ├── trans_sell_analysis.ipynb
│       └── trans_stock_analysis.ipynb
├── img
│   └── MF_pred.png
├── log
│   ├── history-log.json
│   ├── k-log.json
│   ├── log.json
│   └── model_score_log
└── recs
    ├── __init__.py
    ├── backtest.py
    ├── clf.py
    ├── datasets
    │   ├── __init__.py
    │   ├── attribution.py
    │   ├── base.py
    │   └── preprocess.py
    ├── metrics
    │   └── base.py
    ├── models
    │   ├── __init__.py
    │   ├── base.py
    │   ├── mf
    │   │   ├── __init__.py
    │   │   └── mf.py
    │   └── vaecf
    │       ├── __init__.py
    │       ├── recom_vaecf.py
    │       └── vaecf.py
    └── utils
        ├── __init__.py
        ├── common.py
        ├── config.py
        └── path.py
```
## How to inference:
- 在 `./examples/` 中有較簡便的使用示範
### Common
- df 資料格式：
    | user | item| rate |
    |-|-|-|
    | USER A | ITEM 1 | 1 |
    | USER B | ITEM 2 | 1 | 
    | USER B | ITEM 3 | 1 | 
### MF's Inference
- 範例程式碼
  ```python
  # import MF模型
  from recs.models.mf import MatrixFactorization 
  
  # 訓練模型
  MF = MatrixFactorization(df, svds_k = 15)

  # 預測
  u = df.user.tolist()[0]
  MF.rec_items(u)
  ```
- 預測結果示意圖

  ![img](img/MF_pred.png)
  
### VAECF's Inference
- 範例程式碼
  ```python
  import recs

  # 讀取資料給 Dataset
  train_set = recs.datasets.Dataset.from_uir(df.values, seed=35)

  # 初始化模型
  vaecf = recs.models.VAECF(
    k=10,
    autoencoder_structure=[20],
    act_fn="tanh",        # activate function
    likelihood="mult",    # 論文提到效果最好是使用 multinomial likelihood
    n_epochs=30,
    batch_size=30,
    learning_rate=0.001,
    beta=1.0,
    seed=35,
    use_gpu=True,
    verbose=True,
  )

  # 訓練
  vaecf.fit(train_set)

  # 預測
  pred = predict_ranking(vaecf, df, 'userId', 'itemId', 'rank', True)
  ```

## Parameters Details
MF與VAECF模型可調整參數細節寫在 models 資料夾各別模型中