from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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

    datasets = {k: pd.read_csv(v) for k, v in paths.items()}
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
        'lightgbm': LGBMClassifier
    }
    if params:
        return models[model_name](**params)
    else:
        return models[model_name]()

def run_experiment(model_name, dataset_name, params=None, threshold=0.5):
    try:
        data = load_datasets(dataset_name)
        model = get_model(model_name, params)
        model.fit(data['x_train'], data['y_train'])

        metrics = evaluate_model(model, data['x_test'], data['y_test'], threshold=threshold)

        params_str = str(params) if params else "default"

        return {
            'model': model_name,
            'dataset': dataset_name,
            **metrics,
            'params': params_str
        }

    except Exception as e:
        print(f"Error in {model_name}/{dataset_name}: {str(e)}")
        return None

def save_results(results, filename):
    df = pd.DataFrame([r for r in results if r is not None])
    
    if df.empty:
        print("Nenhum resultado válido para salvar!")
        return None
    
    cols_order = ['model', 'dataset', 'f1', 'auc_roc', 'accuracy', 'precision',
                  'recall', 'params', 'threshold_used']
    
    existing_cols = [col for col in cols_order if col in df.columns]
    other_cols = [col for col in df.columns if col not in cols_order]
    final_order = existing_cols + other_cols
    df = df[final_order]

    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", filename)
    df.to_excel(filepath, index=False)
    print(f"\nResults saved to {filepath}")
    return df
