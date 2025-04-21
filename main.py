from preprocessing import preprocessing
from models import run_experiment, save_results
from datetime import datetime

def main():
    
    
    preprocessing()
    
    experiments = [
        {
            "model": "logistic_regression",
            "dataset": "orig",
            "parameters": {
                "C": 0.1,
                "penalty": "l2",
                "class_weight": {0:1, 1:3},
                "max_iter": 3000
            }
        },
        {
            "model": "logistic_regression",
            "dataset": "under",
            "parameters": {
                "C": 0.2,
                "penalty": "l2",
                "class_weight": "balanced",
                "max_iter": 2000
            },
            "threshold": 0.8
        },
        {
            "model": "logistic_regression",
            "dataset": "over",
            "parameters": {
                "C": 0.05,
                "penalty": "l2",
                "class_weight": {0:1, 1:4},
                "max_iter": 5000
            }
        },
        {
            "model": "random_forest",
            "dataset": "orig",
            "parameters": {
                "n_estimators": 300,
                "max_depth": 12,
                "class_weight": {0:1, 1:4}
            }
        },
        {
            "model": "random_forest",
            "dataset": "over",
            "parameters": {
                "n_estimators": 200,
                "max_depth": 8,
                "class_weight": "balanced"
            }
        },
        {
            "model": "xgboost",
            "dataset": "orig",
            "parameters": {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "scale_pos_weight": 3
            }
        }
    ]
    
    all_results = []
    for exp in experiments:
        result = run_experiment(
            model_name=exp["model"],
            dataset_name=exp["dataset"],
            params=exp["parameters"]
        )
        all_results.append(result)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(all_results, f"experiment_results_{timestamp}.xlsx")
    
    print("\nFinished.")

if __name__ == "__main__":
    main()