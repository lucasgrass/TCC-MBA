experiments = [
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
        "threshold": 0.375
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
        "threshold": 0.4
    },
        {
        "model": "random_forest",
        "dataset": "top10_label_category_orig",
        "mode": "single",
        "parameters": {
            "n_estimators": 2000,
            "max_depth": 8,
            "class_weight": {0: 1, 1: 4},
            "max_features": "log2",
            "random_state": 42
        },
        "threshold": 0.375
    },
    {
        "model": "random_forest",
        "dataset": "top10_label_category_orig",
        "mode": "single",
        "parameters": {
            "n_estimators": 2000,
            "max_depth": 10,
            "class_weight": {0: 1, 1: 4},
            "max_features": "log2",
            "random_state": 42
        },
        "threshold": 0.4
    },
        {
        "model": "random_forest",
        "dataset": "top10_fe_label_category_orig",
        "mode": "single",
        "parameters": {
            "n_estimators": 2000,
            "max_depth": 8,
            "class_weight": {0: 1, 1: 4},
            "max_features": "log2",
            "random_state": 42
        },
        "threshold": 0.375
    },
    {
        "model": "random_forest",
        "dataset": "top10_fe_label_category_orig",
        "mode": "single",
        "parameters": {
            "n_estimators": 2000,
            "max_depth": 10,
            "class_weight": {0: 1, 1: 4},
            "max_features": "log2",
            "random_state": 42
        },
        "threshold": 0.4
    },
        {
        "model": "random_forest",
        "dataset": "all_label_category_orig",
        "mode": "single",
        "parameters": {
            "n_estimators": 2000,
            "max_depth": 8,
            "class_weight": {0: 1, 1: 4},
            "max_features": "sqrt",
            "random_state": 42
        },
        "threshold": 0.375
    },
    {
        "model": "random_forest",
        "dataset": "all_label_category_orig",
        "mode": "single",
        "parameters": {
            "n_estimators": 2000,
            "max_depth": 10,
            "class_weight": {0: 1, 1: 4},
            "max_features": "sqrt",
            "random_state": 42
        },
        "threshold": 0.4
    }
]