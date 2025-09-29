experiments = [
    {
        "model": "xgboost",
        "dataset": "all_label_category_under",
        "mode": "single",
        "parameters": {
            "n_estimators": 1500,
            "max_depth": 4,
            "learning_rate": 0.1,
            "scale_pos_weight": 4,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "random_state": 42
        },
        "threshold": 0.375
    },
    {
        "model": "xgboost",
        "dataset": "all_label_category_orig",
        "mode": "single",
        "parameters": {
            "n_estimators": 1500,
            "max_depth": 3,
            "learning_rate": 0.01,
            "scale_pos_weight": 4,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "random_state": 42,
            "colsample_bytree": 0.8,
            "subsample": 0.8
        },
        "threshold": 0.4
    },
]
