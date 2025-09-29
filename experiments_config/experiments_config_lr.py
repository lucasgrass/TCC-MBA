experiments = [
    # {
    #     "model": "logistic_regression",
    #     "dataset": "top10_onehot_scaled_orig",
    #     "mode": "single",
    #     "parameters": {
    #         "solver": "liblinear",
    #         "class_weight": "balanced",
    #         "random_state": 42,
    #         "max_iter": 500
    #     },
    #     "threshold": 0.375
    # },
    # {
    #     "model": "logistic_regression",
    #     "dataset": "top10_onehot_scaled_under",
    #     "mode": "single",
    #     "parameters": {
    #         "solver": "liblinear",
    #         "class_weight": "balanced",
    #         "random_state": 42,
    #         "max_iter": 500
    #     },
    #     "threshold": 0.375
    # },
    # {
    #     "model": "logistic_regression",
    #     "dataset": "top10_onehot_scaled_over",
    #     "mode": "single",
    #     "parameters": {
    #         "solver": "liblinear",
    #         "class_weight": "balanced",
    #         "random_state": 42,
    #         "max_iter": 500
    #     },
    #     "threshold": 0.375
    # },
    # {
    #     "model": "logistic_regression",
    #     "dataset": "top10_fe_onehot_scaled_orig",
    #     "mode": "single",
    #     "parameters": {
    #         "solver": "liblinear",
    #         "class_weight": "balanced",
    #         "random_state": 42,
    #         "max_iter": 500
    #     },
    #     "threshold": 0.375
    # },
    # {
    #     "model": "logistic_regression",
    #     "dataset": "top10_fe_onehot_scaled_under",
    #     "mode": "single",
    #     "parameters": {
    #         "solver": "liblinear",
    #         "class_weight": "balanced",
    #         "random_state": 42,
    #         "max_iter": 500
    #     },
    #     "threshold": 0.375
    # },
    # {
    #     "model": "logistic_regression",
    #     "dataset": "top10_fe_onehot_scaled_over",
    #     "mode": "single",
    #     "parameters": {
    #         "solver": "liblinear",
    #         "class_weight": "balanced",
    #         "random_state": 42,
    #         "max_iter": 500
    #     },
    #     "threshold": 0.375
    # },
    # {
    #     "model": "logistic_regression",
    #     "dataset": "all_onehot_scaled_orig",
    #     "mode": "single",
    #     "parameters": {
    #         "solver": "liblinear",
    #         "class_weight": "balanced",
    #         "random_state": 42,
    #         "max_iter": 500
    #     },
    #     "threshold": 0.375
    # },
    # {
    #     "model": "logistic_regression",
    #     "dataset": "all_onehot_scaled_under",
    #     "mode": "single",
    #     "parameters": {
    #         "solver": "liblinear",
    #         "class_weight": "balanced",
    #         "random_state": 42,
    #         "max_iter": 500
    #     },
    #     "threshold": 0.375
    # },
    # {
    #     "model": "logistic_regression",
    #     "dataset": "all_onehot_scaled_over",
    #     "mode": "single",
    #     "parameters": {
    #         "solver": "liblinear",
    #         "class_weight": "balanced",
    #         "random_state": 42,
    #         "max_iter": 500
    #     },
    #     "threshold": 0.375
    # },
    #     {
    #     "model": "logistic_regression",
    #     "dataset": "top10_onehot_scaled_orig",
    #     "mode": "grid_search",
    #     "parameters_grid": {
    #         "solver": ["sag", "lbfgs"],
    #         "penalty": ["l2"],
    #         "C": [0.01, 0.1, 1, 10],  # regularização
    #         "max_iter": [1000],
    #         "random_state": [42]
    #     },
    #     "thresholds": [0.3, 0.4, 0.5]
    # },
        {
        "model": "logistic_regression",
        "dataset": "top10_fe_onehot_scaled_orig",
        "mode": "single",
        "parameters": {
            "solver": "liblinear",
            "class_weight": {0: 1, 1: 4},
            "random_state": 42,
            "max_iter": 1000
        },
        "threshold": 0.375
    },
    {
        "model": "logistic_regression",
        "dataset": "top10_fe_onehot_scaled_under",
        "mode": "grid_search",
        "parameters_grid": {
            "solver": ["sag", "lbfgs"],
            "penalty": ["l2"],
            "C": [0.01, 0.1, 1, 10],
            "max_iter": [2000],
            "random_state": [42],
            "class_weight": ["balanced"]
        },
        "thresholds": [0.3, 0.4]
    },
    {
    "model": "logistic_regression",
    "dataset": "top10_onehot_scaled_orig",
    "mode": "grid_search",
    "parameters_grid": {
        "solver": ["sag", "lbfgs"],
        "penalty": ["l2"],
        "C": [0.01, 0.1, 1, 10],
        "max_iter": [2000],
        "random_state": [42],
        "class_weight": [{0: 1, 1: 4}]   # <-- sem aspas, dicionário real
    },
    "thresholds": [0.4]
    },
        {
    "model": "logistic_regression",
    "dataset": "top10_onehot_scaled_over",
    "mode": "grid_search",
    "parameters_grid": {
        "solver": ["sag", "lbfgs"],
        "penalty": ["l2"],
        "C": [0.01, 0.1, 1, 10],
        "max_iter": [2000],
        "random_state": [42],
        "class_weight": ["balanced"]   # <-- sem aspas, dicionário real
    },
    "thresholds": [0.4]
    },
        {
    "model": "logistic_regression",
    "dataset": "top10_onehot_scaled_over",
    "mode": "grid_search",
    "parameters_grid": {
        "solver": ["sag", "lbfgs"],
        "penalty": ["l2"],
        "C": [0.01, 0.1, 1, 10],
        "max_iter": [2000],
        "random_state": [42],
        "class_weight": ["balanced"]   # <-- sem aspas, dicionário real
    },
    "thresholds": [0.4]
    },
]
