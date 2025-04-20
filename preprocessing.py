# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    fig, axs = plt.subplots(num_plots, 1, dpi=95, figsize=(7, num_plots * 2))
    
    for i, col in enumerate(numeric_cols):
        axs[i].boxplot(df[col], vert=False)
        axs[i].set_ylabel(col)
    plt.tight_layout()
    plt.show()
    
    print('\nBefore outliers function:', df.shape)
    
    numeric_cols = [col for col in numeric_cols if col != target_column]
    
    for col in numeric_cols:
        df = drop_outliers(df, col)
    
    print('\nAfter outliers function:', df.shape)
    
    print(df['Churn'].value_counts())
    
    # Save cleaned dataset
    df.to_csv(base_path + '/datasets/cleaned_dataset.csv', index=False)
    
    # %% Applying StandardScaler, LabelEncoder and One-Hot Encoder
    
    one_hot_encoder_variables = [
        'PaymentMethod','PaperlessBilling', 'ContentType', 'MultiDeviceAccess',
        'DeviceRegistered', 'GenrePreference', 'Gender', 'ParentalControl',
        'SubtitlesEnabled'
        ]
    
    le = LabelEncoder()
    
    df['SubscriptionType_le'] = le.fit_transform(df['SubscriptionType'])
    
    df = df.drop(columns=['SubscriptionType'])
    
    print(df['SubscriptionType_le'].value_counts())
    
    df = pd.get_dummies(df, columns=one_hot_encoder_variables, drop_first=False)
    
    # %%
    
    x = df.drop(columns=[target_column])
    y = df['Churn']
    
    x_train, x_validation_test, y_train, y_validation_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42)
    
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_validation_test, y_validation_test, test_size=0.5, stratify=y_validation_test, random_state=42
    )
    
    scaler = StandardScaler()
    # %% Padronization in original dataset (unbalanced)
    
    x_train_orig = x_train.copy()
    x_validation_orig = x_validation.copy()
    x_test_orig = x_test.copy()
    
    x_train_orig[numeric_cols] = scaler.fit_transform(x_train_orig[numeric_cols])
    x_validation_orig[numeric_cols] = scaler.transform(x_validation_orig[numeric_cols])
    x_test_orig[numeric_cols] = scaler.transform(x_test_orig[numeric_cols])
    
    # %% Padronization in balanced dataset (Undersampling method)
    
    rus = RandomUnderSampler(random_state=42)
    x_train_under, y_train_under = rus.fit_resample(x_train, y_train)
    
    x_train_under[numeric_cols] = scaler.fit_transform(x_train_under[numeric_cols])
    x_validation_under = x_validation.copy()
    x_test_under = x_test.copy()
    x_validation_under[numeric_cols] = scaler.transform(x_validation_under[numeric_cols])
    x_test_under[numeric_cols] = scaler.transform(x_test_under[numeric_cols])
    
    # %% Padronization in balanced dataset (Oversampling  method)
    
    smote = SMOTE(random_state=42)
    x_train_over, y_train_over = smote.fit_resample(x_train, y_train)
    
    x_train_over[numeric_cols] = scaler.fit_transform(x_train_over[numeric_cols])
    x_validation_over = x_validation.copy()
    x_test_over = x_test.copy()
    x_validation_over[numeric_cols] = scaler.transform(x_validation_over[numeric_cols])
    x_test_over[numeric_cols] = scaler.transform(x_test_over[numeric_cols])
    
    # %% Save the new datasets
    
    x_train_orig.to_csv(base_path + '/datasets/x_train_orig.csv', index=False)
    y_train.to_csv(base_path + '/datasets/y_train_orig.csv', index=False)
    
    x_validation_orig.to_csv(base_path + '/datasets/x_validation_orig.csv', index=False)
    y_validation.to_csv(base_path + '/datasets/y_validation_orig.csv', index=False)
    
    x_test_orig.to_csv(base_path + '/datasets/x_test_orig.csv', index=False)
    y_test.to_csv(base_path + '/datasets/y_test_orig.csv', index=False)
    
    x_train_under.to_csv(base_path + '/datasets/x_train_under.csv', index=False)
    y_train_under.to_csv(base_path + '/datasets/y_train_under.csv', index=False)
    
    x_validation_under.to_csv(base_path + '/datasets/x_validation_under.csv', index=False)
    y_validation.to_csv(base_path + '/datasets/y_validation_under.csv', index=False)
    
    x_test_under.to_csv(base_path + '/datasets/x_test_under.csv', index=False)
    y_test.to_csv(base_path + '/datasets/y_test_under.csv', index=False)
    
    x_validation_over.to_csv(base_path + '/datasets/x_validation_over.csv', index=False)
    y_validation.to_csv(base_path + '/datasets/y_validation_over.csv', index=False)
    
    x_test_over.to_csv(base_path + '/datasets/x_test_over.csv', index=False)
    y_test.to_csv(base_path + '/datasets/y_test_over.csv', index=False)
    
    x_train_over.to_csv(base_path + '/datasets/x_train_over.csv', index=False)
    y_train_over.to_csv(base_path + '/datasets/y_train_over.csv', index=False)



