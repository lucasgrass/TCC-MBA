from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC, LinearSVC
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
            result = train_logistic_regression(dataset)
        elif model == "random_forest":
            result = train_random_forest(dataset)
        elif model == "xgboost":
            result = train_xgboost(dataset)
        elif model == "svm":
            result = train_svm(dataset)
        else:
            print("Error. Use: 'logistic_regression', 'random_forest', 'xgboost' or 'svm'.")
            return

        result["dataset"] = dataset
        result["model"] = model
        all_results.append(result)
        
    df_results = pd.DataFrame(all_results)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path = os.path.join("results", f"model_training_results_{model}_{timestamp}.xlsx")
    df_results.to_excel(file_path, index=False)

    print("\nResults saved.")
    
def run_all_models_save_results():
    all_results = []

    models = [
        ("Logistic Regression", train_logistic_regression),
        ("Random Forest", train_random_forest),
        ("XGBoost", train_xgboost),
        ("SVM", train_svm)
    ]
    
    for dataset in ["orig", "under", "over"]:
        print(f"\nTraining models on {dataset} dataset...")

        for model_name, model_function in models:
            result = model_function(dataset)
            result["dataset"] = dataset
            result["model"] = model_name
            all_results.append(result)
            
    df_results = pd.DataFrame(all_results)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path = os.path.join("results", f"all_model_training_results_{timestamp}.xlsx")
    
    df_results.to_excel(file_path, index=False)

    print(f"\nResults saved to {file_path}.")


def load_datasets(dataset_name):
    x_train_path = f"./datasets/x_train_{dataset_name}.csv"
    y_train_path = f"./datasets/y_train_{dataset_name}.csv"
    x_validation_path = f"./datasets/x_validation_{dataset_name}.csv"
    y_validation_path = f"./datasets/y_validation_{dataset_name}.csv"
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

    y_pred = model.predict(x_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    
    results = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "AUC-ROC": auc_roc
    }
    
    return results

def evaluate_model_comparison_only(model, x_train, x_test, y_train, y_test):

    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy : {test_accuracy:.4f}")

    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    auc_roc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")

    plot_learning_curve(model, x_train, y_train)

    return {
        "Train Accuracy": train_accuracy,
        "Test Accuracy": test_accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "AUC-ROC": auc_roc
    }

def plot_learning_curve(model, x_train, y_train):
    train_sizes, train_scores, test_scores = learning_curve(model, x_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

    plt.plot(train_sizes, train_scores.mean(axis=1), label="Training")
    plt.plot(train_sizes, test_scores.mean(axis=1), label="Test")

    plt.xlabel("Número de amostras de treino")
    plt.ylabel("Pontuação")
    plt.title("Curva de Aprendizado")
    plt.legend()
    plt.show()
    
def train_logistic_regression(base_path_dataset):
    datasets = load_datasets(base_path_dataset)
    
    model = LogisticRegression(max_iter=3000, verbose=1)
    
    model.fit(datasets['x_train'], datasets['y_train'])
    
    results = evaluate_model_comparison_only(model, datasets['x_train'], datasets['x_test'], datasets['y_train'], datasets['y_test'])
    
    #explain_shap(model, datasets['x_train'], datasets['x_test'], model_name='logistic_regression')
    #explain_lime(
    #model=model,
    #x_train=datasets['x_train'],
    #x_test=datasets['x_test'],
    #feature_names=datasets['x_train'].columns.tolist(),
    #class_names=['not_churn', 'churn'],
    #model_name='logistic_regression',
    #)
    
    return results


def train_random_forest(base_path_dataset):
    datasets = load_datasets(base_path_dataset)
    
    model = RandomForestClassifier()
    
    model.fit(datasets['x_train'], datasets['y_train'])

    results = evaluate_model_comparison_only(model, datasets['x_train'], datasets['x_test'], datasets['y_train'], datasets['y_test'])
    return results


def train_xgboost(base_path_dataset):
    datasets = load_datasets(base_path_dataset)

    model = XGBClassifier(
        eval_metric='logloss',
        verbosity=1,
        objective='binary:logistic'
    )

    model.fit(
        datasets["x_train"],
        datasets["y_train"],
        eval_set=[(datasets["x_validation"], datasets["y_validation"])],
        verbose=True
    )

    results = evaluate_model_comparison_only(model, datasets['x_train'], datasets['x_test'], datasets['y_train'], datasets['y_test'])
    return results


def train_svm(base_path_dataset):
    datasets = load_datasets(base_path_dataset)

    model = LinearSVC(max_iter=10000, verbose=True)

    model.fit(datasets['x_train'], datasets['y_train'])

    results = evaluate_model_comparison_only(model, datasets['x_train'], datasets['x_test'], datasets['y_train'], datasets['y_test'])
    return results