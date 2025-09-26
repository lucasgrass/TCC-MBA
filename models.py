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
        'x_train': f"./datasets/x_train_{dataset_name}.csv",
        'y_train': f"./datasets/y_train_{dataset_name}.csv",
        'x_test': f"./datasets/x_test_{dataset_name}.csv",
        'y_test': f"./datasets/y_test_{dataset_name}.csv"
    }
    
    datasets = {k: pd.read_csv(v) for k, v in paths.items()}
    datasets['y_train'] = datasets['y_train'].squeeze()
    datasets['y_test'] = datasets['y_test'].squeeze()
    
    if dataset_name == "clean":
        # Definir colunas categóricas apenas para o dataset clean
        categorical_cols = [
            'SubscriptionType', 'PaymentMethod','PaperlessBilling', 
            'MultiDeviceAccess', 'GenrePreference', 'Gender', 
            'ParentalControl', 'SubtitlesEnabled', 'ContentType', 'DeviceRegistered'
        ]
        
        for col in categorical_cols:
            if col in datasets['x_train'].columns:
                datasets['x_train'][col] = datasets['x_train'][col].astype('category')
                datasets['x_test'][col] = datasets['x_test'][col].astype('category')
    
    return datasets


def evaluate_model(model, x_test, y_test, model_name=None, threshold=None):
    y_prob = model.predict_proba(x_test)[:, 1]
    
    if threshold is None:
        if model_name == 'logistic_regression':
            threshold = 0.5
        elif model_name == 'random_forest':
            threshold = 0.5
        elif model_name == 'xgboost':
            threshold = 0.5
        else:
            threshold = 0.5
    
    y_pred = (y_prob >= threshold).astype(int)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_prob),
        'threshold_used': threshold
    }

def get_model(model_name, params):

    models = {
        'logistic_regression': LogisticRegression,
        'random_forest': RandomForestClassifier,
        'xgboost': XGBClassifier,
        'decision_tree': DecisionTreeClassifier,
        'lightgbm': LGBMClassifier
    }
    return models[model_name](**params)

def run_experiment(model_name, dataset_name, params, threshold=None):
    print(f"\nTraining {model_name} on {dataset_name} dataset with params: {params}")
    
    try:
        data = load_datasets(dataset_name)
        
        model = get_model(model_name, params)
        model.fit(data['x_train'], data['y_train'])
        
        metrics = evaluate_model(
            model=model,
            x_test=data['x_test'],
            y_test=data['y_test'],
            model_name=model_name,
            threshold=threshold
        )
        
        features = None
        if hasattr(model, 'feature_importances_'):
            features = dict(zip(data['x_train'].columns, model.feature_importances_))
        
        return {
            'model': model_name,
            'dataset': dataset_name,
            **metrics,
            'features': features,
            'params': str(params)
        }
        
    except Exception as e:
        print(f"Error in {model_name}/{dataset_name}: {str(e)}")
        return None

def save_results(results, filename):
    df = pd.DataFrame([r for r in results if r is not None])
    
    cols_order = ['model', 'dataset', 'f1', 'auc_roc', 'accuracy', 'precision',
                  'recall', 'params', 'threshold_used', 'features']
    
    existing_cols = [col for col in cols_order if col in df.columns]
    other_cols = [col for col in df.columns if col not in cols_order]
    final_order = existing_cols + other_cols
    
    if final_order:
        df = df[final_order]
    
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", filename)
    df.to_excel(filepath, index=False)
    print(f"\nResults saved to {filepath}")
    return df