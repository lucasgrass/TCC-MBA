# plot_class_distribution_pie_separate.py
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_class_distribution_pie_separate(dataset_paths, dataset_labels=None, target_col='Churn', save_dir="./results"):
    """
    Plota a distribuição da classe target para múltiplos datasets usando gráficos de pizza, salvando cada um separadamente.

    :param dataset_paths: lista de paths dos datasets
    :param dataset_labels: lista de labels correspondentes para cada dataset (opcional)
    :param target_col: nome da coluna target
    :param save_dir: diretório para salvar as figuras
    """
    if dataset_labels is None:
        dataset_labels = [f"Dataset_{i+1}" for i in range(len(dataset_paths))]

    os.makedirs(save_dir, exist_ok=True)

    for path, label in zip(dataset_paths, dataset_labels):
        df = pd.read_csv(path)
        counts = df[target_col].value_counts().sort_index()
        
        plt.figure(figsize=(6, 6))
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'])
        plt.title(f"{label} - Distribuição da classe '{target_col}'")
        
        save_path = os.path.join(save_dir, f"{label}_class_distribution.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Figura salva em: {save_path}")

def print_dataset_sizes(dataset_paths, dataset_labels=None):
    """
    Mostra no terminal o número de linhas de múltiplos datasets.

    :param dataset_paths: lista de paths dos datasets
    :param dataset_labels: lista de labels correspondentes para cada dataset (opcional)
    """
    if dataset_labels is None:
        dataset_labels = [f"Dataset_{i+1}" for i in range(len(dataset_paths))]

    for path, label in zip(dataset_paths, dataset_labels):
        df = pd.read_csv(path)
        print(f"{label}: {len(df)} linhas")


if __name__ == "__main__":
    base_path = "./datasets/all_onehot_scaled"
    datasets = ["orig", "under", "over"]
    paths = [f"{base_path}_{d}/y_train.csv" for d in datasets]
    labels = ["Original", "Undersampling", "Oversampling"]

    #plot_class_distribution_pie_separate(paths, dataset_labels=labels)
    print_dataset_sizes(paths, dataset_labels=labels)