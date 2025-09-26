from preprocessing import preprocessing
from models import run_experiment, save_results
from datetime import datetime

def main():

    preprocessing()
    
    # experiments = [
    #     {
    #         "model": "random_forest",
    #         "dataset": "orig",
    #         "parameters": {
    #             "n_estimators": 300,
    #             "max_depth": 12,
    #             "class_weight": {0:1, 1:4}
    #         }
    #     },
    #     {
    #         "model": "random_forest",
    #         "dataset": "over",
    #         "parameters": {
    #             "n_estimators": 200,
    #             "max_depth": 8,
    #             "class_weight": "balanced"
    #         }
    #     },
    #             {
    #         "model": "xgboost",
    #         "dataset": "orig",
    #         "parameters": {
    #             "n_estimators": 2000,
    #             "max_depth": 6,
    #             "learning_rate": 0.1,
    #             "scale_pos_weight": 3,
    #             "objective":"binary:logistic",
    #             "eval_metric":"aucpr",
    #             "enable_categorical": True
    #         }
    #     },
    #     {
    #         "model": "xgboost",
    #         "dataset": "over",
    #         "parameters": {
    #             "n_estimators": 2000,
    #             "max_depth": 6,
    #             "learning_rate": 0.1,
    #             "scale_pos_weight": 3,
    #             "objective":"binary:logistic",
    #             "eval_metric":"aucpr",
    #             "enable_categorical": True
    #         }
    #     },
    #     {
    #         "model": "xgboost",
    #         "dataset": "clean",
    #         "parameters": {
    #             "n_estimators": 2000,
    #             "max_depth": 6,
    #             "learning_rate": 0.1,
    #             "scale_pos_weight": 3,
    #             "objective":"binary:logistic",
    #             "eval_metric":"aucpr",
    #             "enable_categorical": True
    #         }
    #     },
    #     {
    #         "model": "xgboost",
    #         "dataset": "under",
    #         "parameters": {
    #             "n_estimators": 4000,
    #             "max_depth": 3,
    #             "learning_rate": 0.1,
    #             "scale_pos_weight": 9,
    #             "objective":"binary:logistic",
    #             "eval_metric":"aucpr"
    #         }
    #     },
    #     {
    #         "model": "decision_tree",
    #         "dataset": "orig",
    #         "parameters": {
    #             'ccp_alpha': 0.0,
    #             'class_weight': {0: 1, 1: 3},
    #             'criterion': 'gini',
    #             'max_depth': None,
    #             'max_features': None,
    #             'max_leaf_nodes': None,
    #             'min_impurity_decrease': 0.0,
    #             'min_samples_leaf': 1,
    #             'min_samples_split': 2,
    #             'min_weight_fraction_leaf': 0.0,
    #             'random_state': 45,
    #             'splitter': 'best'
    #         }
    #     },
    #             {
    #         "model": "decision_tree",
    #         "dataset": "over",
    #         "parameters": {
    #             'ccp_alpha': 0.0,
    #             'class_weight': {0: 1, 1: 3},
    #             'criterion': 'gini',
    #             'max_depth': None,
    #             'max_features': None,
    #             'max_leaf_nodes': None,
    #             'min_impurity_decrease': 0.0,
    #             'min_samples_leaf': 1,
    #             'min_samples_split': 2,
    #             'min_weight_fraction_leaf': 0.0,
    #             'random_state': 45,
    #             'splitter': 'best'
    #         }
    #     },
    #             {
    #         "model": "decision_tree",
    #         "dataset": "under",
    #         "parameters": {
    #             'ccp_alpha': 0.0,
    #             'class_weight': {0: 1, 1: 3},
    #             'criterion': 'gini',
    #             'max_depth': None,
    #             'max_features': None,
    #             'max_leaf_nodes': None,
    #             'min_impurity_decrease': 0.0,
    #             'min_samples_leaf': 1,
    #             'min_samples_split': 2,
    #             'min_weight_fraction_leaf': 0.0,
    #             'random_state': 45,
    #             'splitter': 'best'
    #         }
    #     },
    #     {
    #         "model": "lightgbm",
    #         "dataset": "over",
    #         "parameters": {
    #             "n_estimators": 1000,
    #             "max_depth": 6,
    #             "learning_rate": 0.05,
    #             "class_weight": {0:1, 1:3},
    #             "random_state": 42
    #         }
    #     },
    #  {
    #         "model": "lightgbm",
    #         "dataset": "under",
    #         "parameters": {
    #             'boosting_type': 'gbdt',
    #             'class_weight': None,
    #             'colsample_bytree': 1.0,
    #             'importance_type': 'split',
    #             'learning_rate': 0.1,
    #             'max_depth': -1,
    #             'min_child_samples': 20,
    #             'min_child_weight': 0.001,
    #             'min_split_gain': 0.0,
    #             'n_estimators': 2000,
    #             'n_jobs': None,
    #             'num_leaves': 31,
    #             'objective': None,
    #             'random_state': 42,
    #             'reg_alpha': 0.0,
    #             'reg_lambda': 0.0,
    #             'subsample': 1.0,
    #             'subsample_for_bin': 200000,
    #             'subsample_freq': 0,
    #             'verbosity': -1
    #         }
    #     },
    #     {
    #         "model": "lightgbm",
    #         "dataset": "over",
    #         "parameters": {
    #             'boosting_type': 'gbdt',
    #             'class_weight': None,
    #             'colsample_bytree': 1.0,
    #             'importance_type': 'split',
    #             'learning_rate': 0.1,
    #             'max_depth': -1,
    #             'min_child_samples': 20,
    #             'min_child_weight': 0.001,
    #             'min_split_gain': 0.0,
    #             'n_estimators': 1000,
    #             'n_jobs': None,
    #             'num_leaves': 31,
    #             'objective': None,
    #             'random_state': 42,
    #             'reg_alpha': 0.0,
    #             'reg_lambda': 0.0,
    #             'subsample': 1.0,
    #             'subsample_for_bin': 200000,
    #             'subsample_freq': 0,
    #             'verbosity': -1
    #         }
    #     }
    # ]
    
    # all_results = []
    # for exp in experiments:
    #     result = run_experiment(
    #         model_name=exp["model"],
    #         dataset_name=exp["dataset"],
    #         params=exp["parameters"]
    #     )
    #     all_results.append(result)
    
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # save_results(all_results, f"experiment_results_{timestamp}.xlsx")
    
    # print("\nFinished.")

if __name__ == "__main__":
    main()