from preprocessing import preprocessing
from models import run_experiment, save_results
from experiments_config.experiments_config import experiments
from experiments_config.experiments_config_lr import experiments as experiments_lr
from experiments_config.experiments_config_rf import experiments as experiments_rf
from experiments_config.experiments_config_xg import experiments as experiments_xg
from experiments_config.experiments_config_svm import experiments as experiments_svm
from datetime import datetime

def main():
    #preprocessing()

    all_results = []

    for exp in experiments:
        print(f"\nRunning {exp['model']} on {exp['dataset']} (mode={exp.get('mode', 'single')})")
        exp_results = run_experiment(exp)
        all_results.append(exp_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(all_results, f"experiment_results_{timestamp}.xlsx")
    print("\nFinished.")

if __name__ == "__main__":
    main()
