from preprocessing import preprocessing
from models import run_experiment, save_results
from datetime import datetime
from experiments_config import experiments

def main():
    #preprocessing()

    all_results = []

    for exp in experiments:
        print(f"\nRunning {exp['model']} on {exp['dataset']}")
        result = run_experiment(
            model_name=exp["model"], 
            dataset_name=exp["dataset"], 
            params=exp.get("parameters"),
            threshold=exp.get("threshold", 0.5)
        )
        if result is not None:
            all_results.append(result)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(all_results, f"experiment_results_{timestamp}.xlsx")

    print("\nFinished.")


if __name__ == "__main__":
    main()