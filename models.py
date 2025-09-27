from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import ParameterGrid
import pandas as pd
import os

def load_datasets(dataset_name):
    paths = {
        'x_train': f"./datasets/{dataset_name}/x_train.csv",
        'y_train': f"./datasets/{dataset_name}/y_train.csv",
        'x_val': f"./datasets/{dataset_name}/x_val.csv",
        'y_val': f"./datasets/{dataset_name}/y_val.csv",
        'x_test': f"./datasets/{dataset_name}/x_test.csv",
        'y_test': f"./datasets/{dataset_name}/y_test.csv"
    }
    datasets = {k: pd.read_csv(v) for k,v in paths.items()}
    datasets['y_train'] = datasets['y_train'].squeeze()
    datasets['y_val'] = datasets['y_val'].squeeze()
    datasets['y_test'] = datasets['y_test'].squeeze()
    return datasets

def evaluate_model(model, x_test, y_test, threshold=0.5):
    y_prob = model.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_prob),
        'threshold_used': threshold
    }

def get_model(model_name, params=None):
    models = {
        'logistic_regression': LogisticRegression,
        'random_forest': RandomForestClassifier,
        'xgboost': XGBClassifier,
        'decision_tree': DecisionTreeClassifier,
        'lightgbm': LGBMClassifier,
        'svm': SVC
    }
    if params:
        return models[model_name](**params)
    return models[model_name]()

def run_experiment(exp):
    """Roda um experimento single ou grid search e retorna lista de resultados"""
    results = []
    data = load_datasets(exp['dataset'])

    # Decide se é single ou grid
    if exp.get("mode", "single") == "single":
        param_list = [exp.get("parameters", {})]
        thresholds = [exp.get("threshold", 0.5)]
    else:
        param_list = list(ParameterGrid(exp.get("parameters_grid", {})))
        thresholds = exp.get("thresholds", [0.5])

    for params in param_list:
        model = get_model(exp['model'], params)
        model.fit(data['x_train'], data['y_train'])
        for threshold in thresholds:
            metrics = evaluate_model(model, data['x_test'], data['y_test'], threshold)
            results.append({
                'model': exp['model'],
                'dataset': exp['dataset'],
                **metrics,
                'params': str(params)
            })
    return results

def save_results(all_results, filename):
    # flatten list of lists
    flat_results = [item for sublist in all_results for item in sublist if item is not None]
    if not flat_results:
        print("Nenhum resultado válido para salvar!")
        return None
    df = pd.DataFrame(flat_results)
    cols_order = ['model', 'dataset', 'f1', 'auc_roc', 'accuracy', 'precision', 'recall', 'params', 'threshold_used']
    existing_cols = [c for c in cols_order if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_cols]
    df = df[existing_cols + other_cols]

    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", filename)
    df.to_excel(filepath, index=False)
    print(f"\nResults saved to {filepath}")
    return df
