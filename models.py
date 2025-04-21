from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import learning_curve
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from explainability import explain_shap, explain_lime

def run_model_save_results(model):
    all_results = []
    
    for dataset in ["orig", "under", "over"]:
        print(f"\nTraining the '{model}' with dataset '{dataset}'")

        if model == "logistic_regression":
            result_dict = train_logistic_regression(dataset)
        elif model == "random_forest":
            result_dict = train_random_forest(dataset)
        elif model == "xgboost":
            result_dict = train_xgboost(dataset)
        elif model == "svm":
            result_dict = train_svm(dataset)
        else:
            print("Error. Use: 'logistic_regression', 'random_forest', 'xgboost' or 'svm'.")
            return

        result_dict["dataset"] = dataset
        result_dict["model"] = model
        result_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        all_results.append(result_dict)
    
    df_results = pd.DataFrame(all_results)
    
    cols_order = ['timestamp', 'model', 'dataset'] + [c for c in df_results.columns if c not in ['timestamp', 'model', 'dataset']]
    df_results = df_results[cols_order]
    
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join("results", f"{model}_results_{timestamp}.xlsx")
    df_results.to_excel(file_path, index=False)
    
    print(f"\nResults saved to {file_path}")
    return df_results

def run_all_models_save_results():
    all_results = []
    
    models = [
        ("logistic_regression", train_logistic_regression),
        ("random_forest", train_random_forest),
        ("xgboost", train_xgboost),
        ("svm", train_svm)
    ]
    
    for dataset in ["orig", "under", "over"]:
        print(f"\n=== Training models on {dataset} dataset ===")

        for model_name, model_func in models:
            print(f"\nTraining {model_name}...")
            try:
                result_dict = model_func(dataset)
                
                result_dict["dataset"] = dataset
                result_dict["model"] = model_name
                result_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                all_results.append(result_dict)
                print(f"{model_name} completed successfully.")
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
    
    df_results = pd.DataFrame(all_results)
    
    cols_order = ['timestamp', 'model', 'dataset'] + [c for c in df_results.columns if c not in ['timestamp', 'model', 'dataset']]
    df_results = df_results[cols_order]
    
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join("results", f"all_models_results_{timestamp}.xlsx")
    df_results.to_excel(file_path, index=False)
    
    print(f"\nAll results saved to {file_path}")
    return df_results


def load_datasets(dataset_name):
    x_train_path = f"./datasets/x_train_{dataset_name}.csv"
    y_train_path = f"./datasets/y_train_{dataset_name}.csv"
    x_validation_path = f"./datasets/x_val_{dataset_name}.csv"
    y_validation_path = f"./datasets/y_val_{dataset_name}.csv"
    x_test_path = f"./datasets/x_test_{dataset_name}.csv"
    y_test_path = f"./datasets/y_test_{dataset_name}.csv"

    x_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)
    x_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)
    x_validation = pd.read_csv(x_validation_path)
    y_validation = pd.read_csv(y_validation_path)

    y_train = y_train.squeeze()
    y_test = y_test.squeeze()
    y_validation = y_validation.squeeze()
    
    datasets = {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "x_validation": x_validation,
        "y_validation": y_validation
    }
    
    return datasets

def evaluate_model(model, x_train, x_test, y_train, y_test):

    y_pred_test = model.predict(x_test)
    y_prob_test = model.predict_proba(x_test)[:, 1]
    
    y_pred_train = model.predict(x_train)
    y_prob_train = model.predict_proba(x_train)[:, 1]

    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test)
    test_recall = recall_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)
    test_auc_roc = roc_auc_score(y_test, y_prob_test)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    train_auc_roc = roc_auc_score(y_train, y_prob_train)
    
    print("\nTest dataset:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1-Score: {test_f1:.4f}")
    print(f"AUC-ROC: {test_auc_roc:.4f}")
    
    print("\nTrain dataset (overfitting detection):")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Train AUC-ROC: {train_auc_roc:.4f}")
    
    results = {
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_auc_roc": test_auc_roc,
        "train_accuracy": train_accuracy,
        "train_auc_roc": train_auc_roc
    }
    
    return results
    
def train_logistic_regression(base_path_dataset):
    datasets = load_datasets(base_path_dataset)
    
    model_config = {
        'model_type': 'LogisticRegression',
        'parameters': {
            'max_iter': 3000,
            'penalty': 'l2',
            'C': 0.4,
            'class_weight': 'balanced',
            'solver': 'liblinear',
            'random_state': 42
        }
    }
    
    model = LogisticRegression(**model_config['parameters'])
    model.fit(datasets['x_train'], datasets['y_train'])
    
    results = evaluate_model(model, datasets['x_train'], datasets['x_test'], 
                           datasets['y_train'], datasets['y_test'])
    
    return {
        **results,
        **model_config['parameters'],
        'model_type': model_config['model_type'],
        'training_samples': len(datasets['x_train']),
        'test_samples': len(datasets['x_test']),
        'feature_count': datasets['x_train'].shape[1]
    }


def train_random_forest(base_path_dataset):
    datasets = load_datasets(base_path_dataset)
    
    model_config = {
        'model_type': 'RandomForest',
        'parameters': {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
    }
    
    model = RandomForestClassifier(**model_config['parameters'])
    model.fit(datasets['x_train'], datasets['y_train'])
    
    results = evaluate_model(model, datasets['x_train'], datasets['x_test'], 
                           datasets['y_train'], datasets['y_test'])
    
    return {
        **results,
        **model_config['parameters'],
        'model_type': model_config['model_type'],
        'training_samples': len(datasets['x_train']),
        'test_samples': len(datasets['x_test']),
        'feature_count': datasets['x_train'].shape[1],
        'feature_importances': dict(zip(datasets['feature_names'], model.feature_importances_))
    }

def train_xgboost(base_path_dataset):
    datasets = load_datasets(base_path_dataset)
    
    model_config = {
        'model_type': 'XGBoost',
        'parameters': {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'logloss',
            'objective': 'binary:logistic',
            'random_state': 42,
            'early_stopping_rounds': 20,
            'verbosity': 1
        }
    }
    
    model = XGBClassifier(**model_config['parameters'])
    model.fit(
        datasets['x_train'],
        datasets['y_train'],
        eval_set=[(datasets['x_validation'], datasets['y_validation'])],
        verbose=True
    )
    
    results = evaluate_model(model, datasets['x_train'], datasets['x_test'], 
                           datasets['y_train'], datasets['y_test'])
    
    return {
        **results,
        **model_config['parameters'],
        'model_type': model_config['model_type'],
        'training_samples': len(datasets['x_train']),
        'test_samples': len(datasets['x_test']),
        'feature_count': datasets['x_train'].shape[1],
        'best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else None
    }

def train_svm(base_path_dataset):
    datasets = load_datasets(base_path_dataset)
    
    model_config = {
        'model_type': 'SVM',
        'parameters': {
            'C': 0.5,
            'kernel': 'linear',
            'probability': True,
            'class_weight': 'balanced',
            'max_iter': 3000,
            'random_state': 42,
            'verbose': True
        }
    }
    
    model = SVC(**model_config['parameters'])
    model.fit(datasets['x_train'], datasets['y_train'])
    
    results = evaluate_model(model, datasets['x_train'], datasets['x_test'], 
                           datasets['y_train'], datasets['y_test'])
    
    return {
        **results,
        **model_config['parameters'],
        'model_type': model_config['model_type'],
        'training_samples': len(datasets['x_train']),
        'test_samples': len(datasets['x_test']),
        'feature_count': datasets['x_train'].shape[1]
    }