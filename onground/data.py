import os
# import cudf
import pandas as pd

from typing import Union
from env import *


def load_data(X_path: str, y_path: Union[str, None] = None):
    X = pd.read_parquet(X_path)
    print(type(X['customer_ID']))
    X["customer_ID"] = X["customer_ID"].str[-16:].apply(int, base=16)
    X = X.sort_values("S_2")

    if y_path is not None:
        y = pd.read_csv(y_path)
        y["customer_ID"] = y["customer_ID"].str[-16:].apply(int, base=16)
    else:
        y = None

    return X, y


def transform(
    X: pd.DataFrame,
    y: Union[pd.DataFrame, None] = None,
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

    # https://www.kaggle.com/code/junjitakeshima/amex-lgbm-improved-by-new-feature-p2-b9-en-jp
    # X.loc[:, "P2/B9"] = X["P_2"] / X["B_9"]
    # num_features.append("P2/B9")

    # Numerical Feature
    num_df = (
        X[["customer_ID"] + num_features]
        .groupby("customer_ID")[num_features]
        .agg(["first", "mean", "std", "min", "max", "last"])
    )
    num_df.columns = ["_".join(x) for x in num_df.columns]

    for num_feature in num_features:
        num_df.loc[:, f"{num_feature}_diff1"] = (
            num_df[f"{num_feature}_last"] - num_df[f"{num_feature}_mean"]
        )
        num_df.loc[:, f"{num_feature}_diff2"] = (
            num_df[f"{num_feature}_first"] - num_df[f"{num_feature}_mean"]
        )
        num_df.loc[:, f"{num_feature}_lag_sub"] = (
            num_df[f"{num_feature}_last"] - num_df[f"{num_feature}_first"]
        )
        num_df.loc[:, f"{num_feature}_lag_div"] = (
            num_df[f"{num_feature}_last"] / num_df[f"{num_feature}_first"]
        )


    # Categorical Feature
    cat_df = (
        X[["customer_ID"] + cat_features]
        .groupby("customer_ID")[cat_features]
        .agg(["first", "count", "last", "nunique"])
    )
    cat_df.columns = ["_".join(x) for x in cat_df.columns]

    # cnt_df = X.customer_ID.value_counts().to_frame(name="count")

    # Concat Them
    df = pd.concat([num_df, cat_df], axis=1)

    if y is not None:
        y = y.set_index("customer_ID")
        df = df.merge(y, left_index=True, right_index=True, how="left")
        return df.drop(columns="target"), df["target"]
    else:
        return df, None