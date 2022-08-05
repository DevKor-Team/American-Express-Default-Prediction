import os
import wandb
import joblib
import pandas as pd
import numpy as np
import torch

import lightgbm as lgb
from lightgbm import LGBMClassifier, log_evaluation
from sklearn.model_selection import StratifiedKFold
from wandb.lightgbm import wandb_callback, log_summary

from data import load_data, transform
from model import params
from utils import lgb_amex_metric, amex_metric, seed_everything
from env import *
import time

if __name__ == "__main__":
    seed_everything(42)
    wandb.init(
        project="AMEX",
        name="onground",
    )

    start = time.time()

    X, y = load_data(
        X_path=os.path.join(BASE_DIR, "data/train.parquet"),
        y_path=os.path.join(BASE_DIR, "data/train_labels.csv"),
    )
    X, y = transform(X, y)
    # X, y = X.to_pandas(), y.to_pandas()
    print("transformed complete")
    # X.to_parquet(os.path.join(BASE_DIR, "data/X.parquet"))
    # y.to_parquet(os.path.join(BASE_DIR, "data/y.parquet"))
    # print("parquet complete")
    print("time:", time.time() - start)

    kf = StratifiedKFold(n_splits=5)
    for fold, (idx_tr, idx_va) in enumerate(kf.split(X, y)):
        X_tr, X_va = X.iloc[idx_tr], X.iloc[idx_va]
        y_tr, y_va = y.iloc[idx_tr], y.iloc[idx_va]
        break

    lgb_train = lgb.Dataset(X_tr, y_tr)
    lgb_valid = lgb.Dataset(X_va, y_va)

    start = time.time()

    model = lgb.train(
        params=params,
        train_set=lgb_train,
        num_boost_round=3500,
        valid_sets=[lgb_train, lgb_valid],
        feval=lgb_amex_metric,
        callbacks=[
            log_evaluation(100),
            wandb_callback(),
        ],
    )

    print("time:", time.time() - start)

    log_summary(model, save_model_checkpoint=True)

    # Predict validation
    val_pred = model.predict(X_va)
    # Compute fold metric
    score = amex_metric(y_va, val_pred)
    print(f"Our CV score is {score:.4f}")

    joblib.dump(model, f"checkpoints/day2/cv_{score:.4f}.pkl")