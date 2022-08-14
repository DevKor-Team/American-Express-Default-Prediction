import os
import joblib
import pandas as pd

from data import load_data, transform
from utils import seed_everything
from env import *
import time

if __name__ == "__main__":
    seed_everything(42)
    model = joblib.load("checkpoints/day4/cv_0.79796.pkl")

    start = time.time()

    X, _ = load_data(
        X_path=os.path.join(BASE_DIR, "data/test.parquet"),
        y_path=None,
    )
    X, _ = transform(X, None)
    # X = X.to_pandas()
    print(f"[{time.time() - start}] load data complete")

    start = time.time()
    y_pred = model.predict(X)
    print(f"[{time.time() - start}] predict complete")
    print(y_pred.shape)

    start = time.time()
    submission = pd.read_csv(os.path.join(BASE_DIR, "data/sample_submission.csv"))
    submission["cust_ID"] = (
        submission["customer_ID"].apply(lambda x: int(x[-16:], 16)).astype("int64")
    )
    submission[["customer_ID", "cust_ID"]].merge(
        pd.DataFrame(y_pred, columns=["prediction"], index=X.index),
        left_on="cust_ID",
        right_index=True,
        how="left",
    )[["customer_ID", "prediction"]].to_csv("submission.csv", index=False)
    print(f"[{time.time() - start}] submission complete")