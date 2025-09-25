import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.model_params import get_experiment_config, get_param_grid
from new_preprocessing import new_preprocessing
from models import run_experiment, save_results
from datetime import datetime

def run_custom_experiment():
    config = get_experiment_config()
    
    print("CONFIGURAÇÃO DO EXPERIMENTO:")
    print(f"   Modelos:    {config['models']}")
    print(f"   Datasets:   {config['datasets']}")
    print(f"   Parâmetros: {config['param_mode']}")
    print(f"   Iterações:  {config['n_iter']}")
    print("=" * 50)
    
    all_results = []
    
    for model_name in config['models']:
        for dataset_name in config['datasets']:
            print(f"\nExecutando {model_name.upper()} + {dataset_name}")
            
            result = run_experiment(
                model_name=model_name,
                dataset_name=dataset_name,
                param_grid=get_param_grid(model_name),
                search=True,
                n_iter=config['n_iter'],
                threshold=True
            )
            
            if result:
                all_results.append(result)
                print(f"F1: {result['f1']:.3f} | AUC: {result['auc_roc']:.3f}")
            else:
                print(f"Erro na execução de {model_name} com {dataset_name}")
    
    return all_results

def main():
    new_preprocessing()
    
    all_results = run_custom_experiment()
    
    if all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_results(all_results, f"results_{timestamp}.xlsx")
        
        print("\nRANKING FINAL:")
        print("=" * 30)
        for i, result in enumerate(sorted(all_results, key=lambda x: x['f1'], reverse=True)):
            print(f"{i+1:2d}. {result['model']:12} + {result['dataset']:6} → F1: {result['f1']:.3f}")
    else:
        print("Nenhum resultado obtido")

if __name__ == "__main__":
    main()