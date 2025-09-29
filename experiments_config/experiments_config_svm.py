experiments = [
    {
        "model": "svm",
        "dataset": "top10_onehot_scaled_orig",
        "mode": "grid_search",
        "parameters_grid": {
            "kernel": ["linear", "rbf"],
            "C": [0.1, 1, 10],
            "class_weight": [None, "balanced"],
                "probability": [True],
                "random_state": [42]
        },
        "thresholds": [0.3, 0.4]
    },
    {
        "model": "svm",
        "dataset": "top10_fe_onehot_scaled_under",
        "mode": "grid_search",
        "parameters_grid": {
            "kernel": ["rbf", "poly"],
            "C": [1, 10],
            "gamma": ["scale", "auto"],
            "class_weight": ["balanced"],
                "probability": [True],
                "random_state": [42]
        },
        "thresholds": [0.3, 0.5]
    },
    {
        "model": "svm",
        "dataset": "all_onehot_scaled_under",
        "mode": "grid_search",
        "parameters_grid": {
            "kernel": ["linear", "rbf"],
            "C": [0.1, 1],
            "class_weight": ["balanced"],
                "probability": [True],
                "random_state": [42]
        },
        "thresholds": [0.4, 0.5]
    }
]
