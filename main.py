from preprocessing import preprocessing
from models import run_model_save_results, run_all_models_save_results

def main():
    
    #preprocessing()
    
    run_all_models_save_results()
    
    #run_model_save_results('logistic_regression')

if __name__ == "__main__":
    main()