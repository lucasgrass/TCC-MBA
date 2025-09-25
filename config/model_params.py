import numpy as np

MODELS_TO_RUN = ["xgboost", "lightgbm"]
DATASETS_TO_USE = ["over", "orig", "under"]
PARAM_MODE = "detailed"
N_ITERATIONS = 15

XGB_PARAMS = {
    "fast": {
        "n_estimators": [100, 150],
        "max_depth": [3, 4],
        "learning_rate": [0.1],
        "scale_pos_weight": [3, 5]
    },
    "detailed": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 0.9],
        "colsample_bytree": [0.8, 0.9],
        "gamma": [0, 0.1],
        "reg_alpha": [0, 0.1],
        "reg_lambda": [1, 1.5],
        "scale_pos_weight": [3, 5, 7]
    }
}

LGBM_PARAMS = {
    "fast": {
        "n_estimators": [100, 150],
        "max_depth": [3, 4],
        "num_leaves": [15, 20],
        "learning_rate": [0.1],
        "min_child_samples": [20],
        "reg_alpha": [0.3],
        "reg_lambda": [0.3]
    },
    "detailed": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5, 6],
        "num_leaves": [15, 31, 50],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 0.9],
        "colsample_bytree": [0.8, 0.9],
        "reg_alpha": [0, 0.1, 0.3],
        "reg_lambda": [0, 0.5, 1.0]
    }
}

RF_PARAMS = {
    "fast": {
        "n_estimators": [100, 200],
        "max_depth": [10, None],
        "class_weight": ["balanced"]
    },
    "detailed": {
        "n_estimators": [100, 200, 300],
        "max_depth": [8, 10, 12, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
        "class_weight": ["balanced", {0:1, 1:3}]
    }
}

def get_param_grid(model_name):
    """Retorna o grid de parâmetros para um modelo"""
    param_map = {
        "xgboost": XGB_PARAMS,
        "lightgbm": LGBM_PARAMS, 
        "random_forest": RF_PARAMS
    }
    
    if model_name in param_map:
        return param_map[model_name].get(PARAM_MODE, param_map[model_name]["fast"])
    else:
        raise ValueError(f"Modelo {model_name} não configurado")

def get_experiment_config():
    """Retorna a configuração do experimento"""
    return {
        "models": MODELS_TO_RUN,
        "datasets": DATASETS_TO_USE,
        "param_mode": PARAM_MODE,
        "n_iter": N_ITERATIONS
    }