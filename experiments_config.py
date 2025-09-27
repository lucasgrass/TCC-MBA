# all_onehot_orig
# all_onehot_under
# all_onehot_over
# all_label_category_orig
# all_label_category_under
# all_label_category_over
# all_onehot_scaled_orig
# all_onehot_scaled_under
# all_onehot_scaled_over
# top10_onehot_orig
# top10_onehot_under
# top10_onehot_over
# top10_label_category_orig
# top10_label_category_under
# top10_label_category_over
# top10_onehot_scaled_orig
# top10_onehot_scaled_under
# top10_onehot_scaled_over
# top10_fe_onehot_orig
# top10_fe_onehot_under
# top10_fe_onehot_over
# top10_fe_label_category_orig
# top10_fe_label_category_under
# top10_fe_label_category_over
# top10_fe_onehot_scaled_orig
# top10_fe_onehot_scaled_under
# top10_fe_onehot_scaled_over

experiments = [
    # {
    #     "model": "random_forest",
    #     "dataset": "top10_fe_label_category_over",
    #     "mode": "single",
    #     "parameters": {
    #         "n_estimators": 200,
    #         "max_depth": 4,
    #         "class_weight": "balanced",
    #         "random_state": 42
    #     },
    #     "threshold": 0.3
    # },
    # {
    #     "model": "random_forest",
    #     "dataset": "top10_fe_label_category_over",
    #     "mode": "single",
    #     "parameters": {
    #         "n_estimators": 300,
    #         "max_depth": 4,
    #         "class_weight": "balanced",
    #         "random_state": 42
    #     },
    #     "threshold": 0.3
    # },
    # {
    #     "model": "random_forest",
    #     "dataset": "top10_fe_label_category_over",
    #     "mode": "single",
    #     "parameters": {
    #         "n_estimators": 400,
    #         "max_depth": 4,
    #         "class_weight": "balanced",
    #         "random_state": 42
    #     },
    #     "threshold": 0.3
    # },
    # {
    #     "model": "decision_tree",
    #     "dataset": "top10_fe_onehot_scaled_under",
    #     "mode": "single",
    #     "parameters": {
    #         "max_depth": 4,
    #         "class_weight": "balanced",
    #         "random_state": 42
    #     },
    #     "threshold": 0.4
    # },
    {
        "model": "xgboost",
        "dataset": "top10_label_category_orig",
        "mode": "single",
        "parameters": {
            "n_estimators": 2000,
            "max_depth": 4,
            "learning_rate": 0.1,
            "scale_pos_weight": 4,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "random_state": 42
        },
        "threshold": 0.4
    },
        {
        "model": "xgboost",
        "dataset": "top10_label_category_over",
        "mode": "single",
        "parameters": {
            "n_estimators": 2000,
            "max_depth": 4,
            "learning_rate": 0.1,
            "scale_pos_weight": 4,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "random_state": 42
        },
        "threshold": 0.4
    },
        {
        "model": "xgboost",
        "dataset": "all_label_category_orig",
        "mode": "single",
        "parameters": {
            "n_estimators": 1000,
            "max_depth": 4,
            "learning_rate": 0.1,
            "scale_pos_weight": 4,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "random_state": 42
        },
        "threshold": 0.35
    },
        {
        "model": "xgboost",
        "dataset": "all_label_category_over",
        "mode": "single",
        "parameters": {
            "n_estimators": 1000,
            "max_depth": 4,
            "learning_rate": 0.1,
            "scale_pos_weight": 4,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "random_state": 42
        },
        "threshold": 0.35
    },
    # {
    #     "model": "lightgbm",
    #     "dataset": "top10_fe_onehot_scaled_under",
    #     "mode": "single",
    #     "parameters": {
    #         "num_leaves": 31,
    #         "max_depth": -1,
    #         "learning_rate": 0.05,
    #         "n_estimators": 100,
    #         "scale_pos_weight": 4,
    #         "random_state": 42
    #     },
    #     "threshold": 0.4
    # }
]
