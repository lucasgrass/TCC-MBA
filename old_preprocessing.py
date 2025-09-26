# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from utils import drop_outliers

# %%
def preprocessing():
        
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    target_column = 'Churn'
    
    df = pd.read_csv(os.path.join(base_path, 'datasets/raw.csv'), sep=',', decimal='.')
    
    print(df['Churn'].value_counts())
    
    #  Unique identifiers do not impact model predictions 
    df.drop('CustomerID', axis=1, inplace=True)
    
    # Check if there is null values
    print('\n null values:', df.isnull().sum())
    df.dropna(inplace=True)
    
    #  Check if there is duplicate lines
    print('\nDuplicated lines:', df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    
    
    # Removing outliers
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    num_plots = len(numeric_cols)
    # fig, axs = plt.subplots(num_plots, 1, dpi=95, figsize=(7, num_plots * 2))
    
    # for i, col in enumerate(numeric_cols):
    #     axs[i].boxplot(df[col], vert=False)
    #     axs[i].set_ylabel(col)
    # plt.tight_layout()
    # plt.show()
    
    print('\nBefore outliers function:', df.shape)
    
    numeric_cols = [col for col in numeric_cols if col != target_column]
    
    for col in numeric_cols:
        df = drop_outliers(df, col)
    
    print('\nAfter outliers function:', df.shape)
    
    print(df['Churn'].value_counts())

    # %% Save cleaned dataset as X and y before encoding/scaling
    # Convert categorical columns to category dtype
    categorical_cols = [
        'SubscriptionType', 'PaymentMethod','PaperlessBilling', 
        'MultiDeviceAccess', 'GenrePreference', 'Gender', 
        'ParentalControl', 'SubtitlesEnabled', 'ContentType', 'DeviceRegistered'
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Separando X e y
    x_clean = df.drop(columns=[target_column])
    y_clean = df[target_column]

    # Dividindo em treino e teste 80/20
    x_train_clean, x_test_clean, y_train_clean, y_test_clean = train_test_split(
        x_clean, y_clean, test_size=0.2, stratify=y_clean, random_state=42
    )

    # Salvando
    x_train_clean.to_csv(os.path.join(base_path, 'datasets/x_train_clean.csv'), index=False)
    x_test_clean.to_csv(os.path.join(base_path, 'datasets/x_test_clean.csv'), index=False)
    y_train_clean.to_csv(os.path.join(base_path, 'datasets/y_train_clean.csv'), index=False)
    y_test_clean.to_csv(os.path.join(base_path, 'datasets/y_test_clean.csv'), index=False)

    print("Clean dataset saved: X and y, 80/20 split, without encoding/scaling")

    
    # Save cleaned dataset
    df.to_csv(base_path + '/datasets/cleaned_dataset.csv', index=False)
    
    # %% Applying StandardScaler, LabelEncoder and One-Hot Encoder
    
    one_hot_encoder_variables = [
        'PaymentMethod','PaperlessBilling', 
        'MultiDeviceAccess', 'GenrePreference', 'Gender', 
        'ParentalControl', 'SubtitlesEnabled', 'ContentType', 'DeviceRegistered'
        ]
    
    label_encoder = LabelEncoder()
    df['SubscriptionType'] = label_encoder.fit_transform(df['SubscriptionType'])
    
    df = pd.get_dummies(df, columns=one_hot_encoder_variables, drop_first=False)
    
    # %%
    
    x = df.drop(columns=[target_column])
    y = df['Churn']
    
    x_train, x_val_test, y_train, y_val_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42)
    
    x_val, x_test, y_val, y_test = train_test_split(
        x_val_test, y_val_test, test_size=0.5, stratify=y_val_test, random_state=42
    )
    
    # %% Padronization in original dataset (unbalanced)
    
    x_train_orig = x_train.copy()
    x_val_orig = x_val.copy()
    x_test_orig = x_test.copy()
    
    scaler = StandardScaler()
    
    x_train_orig[numeric_cols] = scaler.fit_transform(x_train_orig[numeric_cols])
    x_val_orig[numeric_cols] = scaler.transform(x_val_orig[numeric_cols])
    x_test_orig[numeric_cols] = scaler.transform(x_test_orig[numeric_cols])
    
    # %% Padronization in balanced dataset (Undersampling method)
    
    rus = RandomUnderSampler(random_state=42)
    x_train_under, y_train_under = rus.fit_resample(x_train_orig, y_train)
    
    # %% Padronization in balanced dataset (Oversampling  method)
    
    smote = SMOTE(random_state=42, k_neighbors=5)
    x_train_over, y_train_over = smote.fit_resample(x_train_orig, y_train)
    
    # %% Save the new datasets
    
    # Original
    pd.DataFrame(x_train_orig).to_csv(f'{base_path}/datasets/x_train_orig.csv', index=False)
    y_train.to_csv(f'{base_path}/datasets/y_train_orig.csv', index=False)
    
    pd.DataFrame(x_val_orig).to_csv(f'{base_path}/datasets/x_val_orig.csv', index=False)
    y_val.to_csv(f'{base_path}/datasets/y_val_orig.csv', index=False)
    
    pd.DataFrame(x_test_orig).to_csv(f'{base_path}/datasets/x_test_orig.csv', index=False)
    y_test.to_csv(f'{base_path}/datasets/y_test_orig.csv', index=False)
    
    # Undersampled
    pd.DataFrame(x_train_under).to_csv(f'{base_path}/datasets/x_train_under.csv', index=False)
    pd.Series(y_train_under).to_csv(f'{base_path}/datasets/y_train_under.csv', index=False)
    
    pd.DataFrame(x_val_orig).to_csv(f'{base_path}/datasets/x_val_under.csv', index=False)
    y_val.to_csv(f'{base_path}/datasets/y_val_under.csv', index=False)
    
    pd.DataFrame(x_test_orig).to_csv(f'{base_path}/datasets/x_test_under.csv', index=False)
    y_test.to_csv(f'{base_path}/datasets/y_test_under.csv', index=False)
    
    # Oversampled
    pd.DataFrame(x_train_over).to_csv(f'{base_path}/datasets/x_train_over.csv', index=False)
    pd.Series(y_train_over).to_csv(f'{base_path}/datasets/y_train_over.csv', index=False)
    
    pd.DataFrame(x_val_orig).to_csv(f'{base_path}/datasets/x_val_over.csv', index=False)
    y_val.to_csv(f'{base_path}/datasets/y_val_over.csv', index=False)
    
    pd.DataFrame(x_test_orig).to_csv(f'{base_path}/datasets/x_test_over.csv', index=False)
    y_test.to_csv(f'{base_path}/datasets/y_test_over.csv', index=False)
    
    print(f"All datasets saved in: {base_path}")

