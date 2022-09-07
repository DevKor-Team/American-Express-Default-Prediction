import os
import cudf
import joblib
import pandas as pd

from data import load_data, transform
from utils import seed_everything
from env import *

if __name__ == "__main__":
    seed_everything(42)
    model = joblib.load("./checkpoints/day2/cv_0.7954.pkl")

    X, _ = load_data(
        X_path=os.path.join(BASE_DIR, "raddar/test.parquet"),
        y_path=None,
    )
    X, _ = transform(X, None)
    X = X.to_pandas()

    y_pred = model.predict(X)
    print(y_pred.shape)

    submission = pd.read_csv(os.path.join(BASE_DIR, "raw/sample_submission.csv"))
    submission["cust_ID"] = (
        submission["customer_ID"].apply(lambda x: int(x[-16:], 16)).astype("int64")
    )
    submission[["customer_ID", "cust_ID"]].merge(
        pd.DataFrame(y_pred, columns=["prediction"], index=X.index),
        left_on="cust_ID",
        right_index=True,
        how="left",
    )[["customer_ID", "prediction"]].to_csv("submission.csv", index=False)
