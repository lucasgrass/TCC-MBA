from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def hyperparam_search(model_class, param_grid, x_train, y_train, scoring="f1", cv=5, n_iter=50):
    """Versão mais robusta"""
    # Calcula o número real de combinações
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
    
    # Ajusta n_iter se necessário
    actual_n_iter = min(n_iter, total_combinations)
    if actual_n_iter < n_iter:
        print(f"⚠️  Ajustando n_iter de {n_iter} para {actual_n_iter} (total de combinações)")
    
    search = RandomizedSearchCV(
        estimator=model_class(),
        param_distributions=param_grid,
        n_iter=actual_n_iter,
        scoring=scoring,
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # Adiciona tratamento de erro
    try:
        search.fit(x_train, y_train)
        return search.best_params_, search.best_score_
    except Exception as e:
        print(f"❌ Erro no hyperparameter search: {e}")
        # Retorna parâmetros default
        return {}, 0

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
    
    return datasets

def evaluate_model(model, x_test, y_test, model_name=None, threshold=None, optimize_threshold=True):
    """
    Avalia o modelo, retornando métricas e threshold usado.
    Se optimize_threshold=True, encontra o threshold que maximiza o F1 score.
    """
    # Obter probabilidades da classe positiva
    y_prob = model.predict_proba(x_test)[:, 1]

    # Determinar threshold automaticamente
    if optimize_threshold:
        precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  # evita divisão por zero
        best_idx = np.argmax(f1_scores)
        threshold = thresholds[best_idx]
    else:
        if threshold is None:
            if model_name == 'logistic_regression':
                threshold = 0.4
            elif model_name in ['random_forest', 'xgboost']:
                threshold = 0.45
            else:
                threshold = 0.5

    # Aplicar threshold
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
        'svm': SVC,
        'lightgbm': LGBMClassifier
    }
    return models[model_name](**params)

def run_experiment(model_name, dataset_name, params=None, threshold=None, param_grid=None, search=False, n_iter=30):
    print(f"\nTraining {model_name} on {dataset_name} dataset")

    try:
        data = load_datasets(dataset_name)

        model_class = {
            'logistic_regression': LogisticRegression,
            'random_forest': RandomForestClassifier,
            'xgboost': XGBClassifier,
            'svm': SVC,
            'lightgbm': LGBMClassifier  # ⬅️ ADICIONAR ESTA LINHA
        }[model_name]  # ⬅️ ESTÁ FALTANDO lightgbm AQUI

        best_params = params
        best_score = None

        if search and param_grid:
            print(f" --> Running hyperparameter search for {model_name}")
            best_params, best_score = hyperparam_search(
                model_class, param_grid, data['x_train'], data['y_train'], n_iter=n_iter
            )
            print(f" --> Best params: {best_params}, best CV score: {best_score:.4f}")

        model = model_class(**best_params)
        model.fit(data['x_train'], data['y_train'])

        if hasattr(model, 'feature_importances_'):
            print(f"\n🎯 ANALYZING FEATURE IMPORTANCE FOR {model_name.upper()}")
            feature_names = data['x_train'].columns.tolist()
            
            # Chama a função de análise
            importance_df = analyze_feature_importance(
                model=model,
                feature_names=feature_names,
                top_n=15
            )
            
            # Salva as top features no resultado
            top_features = importance_df.head(10)[['feature', 'importance']].to_dict('records') if importance_df is not None else None
        else:
            top_features = None

        metrics = evaluate_model(
            model=model,
            x_test=data['x_test'],
            y_test=data['y_test'],
            model_name=model_name,
            optimize_threshold=True
        )

        return {
            'model': model_name,
            'dataset': dataset_name,
            **metrics,
            'top_features': top_features,
            'params': str(best_params),
            'cv_best_score': best_score
        }

    except Exception as e:
        print(f"Error in {model_name}/{dataset_name}: {str(e)}")
        return None
    
def analyze_feature_importance(model, feature_names, top_n=15, correlation_series=None):
    """Analisa a importância real das features após o modelo"""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔍 FEATURE IMPORTANCE DO MODELO:")
        print(importance_df.head(top_n))
        
        # Plot da importância
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(top_n).sort_values('importance', ascending=True)
        plt.barh(top_features['feature'], top_features['importance'])
        plt.title(f'Top {top_n} Features - Importance do Modelo')
        plt.tight_layout()
        plt.show()
        
        # Comparar com correlação se for fornecida
        if correlation_series is not None:
            correlation_df = correlation_series.reset_index()
            correlation_df.columns = ['feature', 'correlation']
            
            comparison_df = importance_df.merge(
                correlation_df, on='feature', how='left'
            )
            comparison_df = comparison_df.sort_values('importance', ascending=False)
            
            print("\n📊 CORRELAÇÃO vs IMPORTÂNCIA (Top 10):")
            print(comparison_df[['feature', 'importance', 'correlation']].head(10))
            
            # Plot comparativo
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Importance
            top_comp = comparison_df.head(10).sort_values('importance', ascending=True)
            ax1.barh(top_comp['feature'], top_comp['importance'])
            ax1.set_title('Top 10 - Importance do Modelo')
            
            # Correlation
            top_comp_corr = comparison_df.head(10).sort_values('correlation', ascending=True)
            ax2.barh(top_comp_corr['feature'], top_comp_corr['correlation'])
            ax2.set_title('Top 10 - Correlação Linear')
            
            plt.tight_layout()
            plt.show()
        
        return importance_df
    else:
        print("⚠️ Modelo não tem feature_importances_")
        return None

def save_results(results, filename):
    df = pd.DataFrame([r for r in results if r is not None])
    
    cols_order = ['model', 'dataset', 'f1', 'auc_roc', 'accuracy', 'precision',
                  'recall', 'params', 'threshold_used', 'features', 'cv_best_score']
    
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
