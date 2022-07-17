from lightgbm import LGBMClassifier

# model = LGBMClassifier(
#     boosting_type="dart",
#     n_estimators=4800,
#     learning_rate=0.01,
#     reg_lambda=50,
#     min_child_samples=2400,
#     num_leaves=95,  # with cpu 95
#     colsample_bytree=0.19,
#     max_bins=511,  # originally for CPU 511, for gpu 255
#     random_state=42,
#     n_jobs=16,
#     # device="gpu",
#     # gpu_platform_id=0,
#     # gpu_device_id=0,
# )
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting": "dart",
    "seed": 42,
    "num_leaves": 100,
    "learning_rate": 0.01,
    "feature_fraction": 0.20,
    "bagging_freq": 10,
    "bagging_fraction": 0.50,
    "n_jobs": -1,
    "lambda_l2": 2,
    "min_data_in_leaf": 40,
}
