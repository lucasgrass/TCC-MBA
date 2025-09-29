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

# all_label_category_orig
# all_label_category_under
# all_label_category_over
# all_onehot_scaled_orig
# all_onehot_scaled_under
# all_onehot_scaled_over
# top10_label_category_orig
# top10_label_category_under
# top10_label_category_over
# top10_onehot_scaled_orig
# top10_onehot_scaled_under
# top10_onehot_scaled_over
# top10_fe_label_category_orig
# top10_fe_label_category_under
# top10_fe_label_category_over
# top10_fe_onehot_scaled_orig
# top10_fe_onehot_scaled_under
# top10_fe_onehot_scaled_over

experiments = [
    {
        "model": "logistic_regression",
        "dataset": "all_onehot_scaled_orig",
        "mode": "single",
        "parameters": {
            "solver": "liblinear",
            "class_weight": {0: 1, 1: 5},
            "random_state": 42,
            "max_iter": 2000
        },
        "threshold": 0.5
    },
    {
        "model": "logistic_regression",
        "dataset": "all_onehot_scaled_under",
        "mode": "single",
        "parameters": {
            "C": 10,
            "class_weight": "balanced",
            "max_iter": 2000,
            "penalty": "l2",
            "random_state": 42,
            "solver": "sag"
        },
        "threshold": 0.5
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
        "threshold": 0.5
    },
    {
        "model": "xgboost",
        "dataset": "all_label_category_orig",
        "mode": "single",
        "parameters": {
            "n_estimators": 2000,
            "max_depth": 3,
            "learning_rate": 0.01,
            "scale_pos_weight": 5,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "random_state": 42,
            "colsample_bytree": 0.8,
            "subsample": 0.8
        },
        "threshold": 0.5
    },
    {
        "model": "random_forest",
        "dataset": "all_label_category_orig",
        "mode": "single",
        "parameters": {
            "n_estimators": 2000,
            "max_depth": 8,
            "class_weight": {0: 1, 1: 4},
            "max_features": "log2",
            "random_state": 42
        },
        "threshold": 0.5
    },
    {
        "model": "random_forest",
        "dataset": "all_label_category_orig",
        "mode": "single",
        "parameters": {
            "n_estimators": 2000,
            "max_depth": 10,
            "class_weight": {0: 1, 1: 4},
            "max_features": "log2",
            "random_state": 42
        },
        "threshold": 0.5
    },
    {
        "model": "svm",
        "dataset": "all_onehot_scaled_under",
        "mode": "single",
        "parameters": {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "class_weight": "balanced",
            "probability": True,
            "random_state": 42
        },
        "threshold": 0.5
    },
    {
        "model": "svm",
        "dataset": "top10_onehot_scaled_under",
        "mode": "single",
        "parameters": {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "class_weight": "balanced",
            "probability": True,
            "random_state": 42
        },
        "threshold": 0.5
    }
]
