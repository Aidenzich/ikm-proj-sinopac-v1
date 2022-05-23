# Sinopac Recsys
## Repository Structure
```
.
├── .gitignore
├── README.md
├── examples
│   ├── mf_example.py
│   └── vaecf_example.py
├── experience
│   ├── cluster
│   ├── collaborative_filtering
│   └── others
├── img
│   └── MF_pred.png
└── recs
    ├── __init__.py
    ├── backtest.py
    ├── datasets
    │   ├── __init__.py
    │   ├── base.py
    │   └── preprocess.py
    ├── metrics
    │   └── base.py
    ├── models
    │   ├── __init__.py
    │   ├── base.py
    │   ├── cluster
    │   │   ├── __init__.py
    │   │   └── clustering.py
    │   ├── explain
    │   │   └── explain.py
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