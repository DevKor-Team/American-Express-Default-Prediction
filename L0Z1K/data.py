import os
import cudf
import pandas as pd

from typing import Union
from env import *


def load_data(X_path: str, y_path: Union[str, None] = None):
    X = cudf.read_parquet(X_path)
    X["customer_ID"] = X["customer_ID"].str[-16:].str.hex_to_int().astype("int64")
    X = X.sort_values("S_2")

    if y_path is not None:
        y = cudf.read_csv(y_path)
        y["customer_ID"] = y["customer_ID"].str[-16:].str.hex_to_int().astype("int64")
    else:
        y = None

    return X, y


def transform(
    X: cudf.DataFrame,
    y: Union[cudf.DataFrame, None] = None,
):
    all_cols = [c for c in list(X.columns) if c not in ["customer_ID", "S_2"]]
    cat_features = [
        "B_30",
        "B_38",
        "D_114",
        "D_116",
        "D_117",
        "D_120",
        "D_126",
        "D_63",
        "D_64",
        "D_66",
        "D_68",
    ]
    num_features = [col for col in all_cols if col not in cat_features]

    # Numerical Feature
    num_df = (
        X[["customer_ID"] + num_features]
        .groupby("customer_ID")[num_features]
        .agg(["mean", "std", "min", "max", "last"])
    )
    num_df.columns = ["_".join(x) for x in num_df.columns]

    # Categorical Feature
    cat_df = (
        X[["customer_ID"] + cat_features]
        .groupby("customer_ID")[cat_features]
        .agg(["count", "last", "nunique"])
    )
    cat_df.columns = ["_".join(x) for x in cat_df.columns]

    # Concat Them
    df = cudf.concat([num_df, cat_df], axis=1)

    if y is not None:
        y = y.set_index("customer_ID")
        df = df.merge(y, left_index=True, right_index=True, how="left")
        return df.drop(columns="target"), df["target"]
    else:
        return df, None
