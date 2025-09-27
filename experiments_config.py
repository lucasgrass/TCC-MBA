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

# experiments_config.py

experiments = [
    {
        "model": "random_forest",
        "dataset": "all_onehot_scaled_over",
        "mode": "single",
        "parameters": {
            "n_estimators": 500,
            "max_depth": 6,
            "class_weight": {0:1, 1:4},
            "random_state": 42
        },
        "threshold": 0.5
    },
    {
        "model": "random_forest",
        "dataset": "top10_fe_label_category_over",
        "mode": "grid_search",
        "parameters_grid": {
            "n_estimators": [200, 500, 1000],
            "max_depth": [4, 6, 8],
            "class_weight": [{0:1,1:1}, {0:1,1:4}]
        },
        "thresholds": [0.4, 0.5, 0.6]
    },
    {
        "model": "xgboost",
        "dataset": "top10_fe_label_category_over",
        "mode": "grid_search",
        "parameters_grid": {
            "n_estimators": [500, 1000, 2000],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1, 0.2],
            "scale_pos_weight": [1, 5],
            "objective": ["binary:logistic"],
            "eval_metric": ["aucpr"]
        },
        "thresholds": [0.4, 0.5, 0.6]
    },
    {
        "model": "lightgbm",
        "dataset": "top10_fe_label_category_over",
        "mode": "grid_search",
        "parameters_grid": {
            "n_estimators": [500, 1000],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1],
            "class_weight": [None, "balanced"]
        },
        "thresholds": [0.5]
    },
    {
        "model": "logistic_regression",
        "dataset": "all_onehot_scaled_over",
        "mode": "grid_search",
        "parameters_grid": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"]
        },
        "thresholds": [0.5]
    },
    {
        "model": "svm",
        "dataset": "all_onehot_scaled_over",
        "mode": "grid_search",
        "parameters_grid": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "probability": [True]
        },
        "thresholds": [0.5]
    }
]


