import shap
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from lime.lime_tabular import LimeTabularExplainer

def explain_shap(model, x_train, x_test, model_name, sample_size=100):
    print(f'\nSHAP: {model_name}')
    
    # Certifique-se de que x_train e x_test estão em numpy arrays
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    
    
    if model_name == 'logistic_regression':
        # Usar predict_proba para garantir que o modelo retorne as probabilidades
        explainer = shap.Explainer(model, x_train)
        shap_values = explainer(x_test)
        print(shap_values)
        print(x_test)
        shap.summary_plot(shap_values, x_test)
    
    elif model_name in ['random_forest', 'xgboost']:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_test)
        shap.summary_plot(shap_values, x_test)
    
    elif model_name == 'svm':
        print("⚠️ Usando KernelExplainer para SVM - pode ser lento, reduzindo amostra...")
        sample_x = shap.sample(x_train, sample_size)
        explainer = shap.KernelExplainer(model.predict_proba, sample_x)
        shap_values = explainer.shap_values(x_test[:50])
        shap.summary_plot(shap_values[1], x_test[:50])

    else:
        raise ValueError("Error. Use: 'logistic_regression', 'random_forest', 'xgboost', ou 'svm'.")


def explain_lime(model, x_train, x_test, feature_names, class_names, model_name, sample_size=3):
    print(f'\nLIME: {model_name}')
    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    
    explainer = LimeTabularExplainer(
        training_data=x_train,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )

    for i in range(sample_size):
        exp = explainer.explain_instance(
            data_row=x_test[i],
            predict_fn=model.predict_proba,
            num_features=10
        )
        print(f'\nExplicação para a amostra {i}:')
        exp.show_in_notebook(show_table=True)

