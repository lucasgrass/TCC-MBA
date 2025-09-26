# %%
import os
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from utils import drop_outliers, onehot_encode, label_encode, label_encode_with_category_simple, scale_numeric, train_val_test_split

def load_and_clean_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base_path, 'datasets/raw.csv'), sep=',', decimal='.')
    
    df.drop('CustomerID', axis=1, inplace=True)
    df.dropna(inplace=True)
    df = df.drop_duplicates().reset_index(drop=True)

    numeric_cols = [
      'AccountAge',
      'MonthlyCharges',
      'TotalCharges',
      'ViewingHoursPerWeek',
      'AverageViewingDuration',
      'ContentDownloadsPerMonth',
      'UserRating',
      'SupportTicketsPerMonth',
      'WatchlistSize'
    ]

    print('\nBefore outliers function:', df.shape)
  
    for col in numeric_cols:
        df = drop_outliers(df, col)
    
    print('\nAfter outliers function:', df.shape)
    
    return df, base_path

def balance_dataset(X_train, y_train):
    rus = RandomUnderSampler(random_state=42)
    smote = SMOTE(random_state=42)
    
    X_train_under, y_train_under = rus.fit_resample(X_train, y_train)
    X_train_over, y_train_over = smote.fit_resample(X_train, y_train)
    
    return {
        'orig': (X_train, y_train),
        'under': (X_train_under, y_train_under),
        'over': (X_train_over, y_train_over)
    }

def create_all_features_datasets(df, base_path):
    """Versão otimizada - mais eficiente e limpa"""
    target_col = 'Churn'
    feat_name = 'all'

    numeric_cols = [
        'AccountAge', 'MonthlyCharges', 'TotalCharges', 'ViewingHoursPerWeek',
        'AverageViewingDuration', 'ContentDownloadsPerMonth', 'UserRating',
        'SupportTicketsPerMonth', 'WatchlistSize'
    ]

    categorical_cols = [
        'SubscriptionType', 'PaymentMethod', 'PaperlessBilling', 'ContentType',
        'MultiDeviceAccess', 'DeviceRegistered', 'GenrePreference', 'Gender',
        'ParentalControl', 'SubtitlesEnabled'
    ]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    # 1) one-hot
    X_tr_oh, X_val_oh, X_test_oh = onehot_encode(X_train, X_val, X_test, categorical_cols)
    
    # One-hot w/o scaling
    save_dataset(X_tr_oh, X_val_oh, X_test_oh, y_train, y_val, y_test,
                 base_path, feat_name, 'orig', 'onehot')

    # One-hot with scaling
    X_tr_scaled, X_val_scaled, X_test_scaled = scale_numeric(X_tr_oh, X_val_oh, X_test_oh, numeric_cols)
    save_dataset(X_tr_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test,
                 base_path, feat_name, 'orig', 'onehot_scaled')

    # 2) Label encoding + category w/o scaling
    X_tr_le, X_val_le, X_test_le = label_encode_with_category_simple(X_train, X_val, X_test, categorical_cols)
    save_dataset(X_tr_le, X_val_le, X_test_le, y_train, y_val, y_test,
                 base_path, feat_name, 'orig', 'label_category')


def create_top10_datasets(df, base_path):

    target_col = 'Churn'
    feat_name = 'top10'
      
    top10_features = [
        'AccountAge', 'MonthlyCharges', 'TotalCharges', 'SubscriptionType',
        'DeviceRegistered', 'ViewingHoursPerWeek', 'ContentDownloadsPerMonth',
        'UserRating', 'SupportTicketsPerMonth', 'WatchlistSize'
    ]

    top10_numeric = [
        'AccountAge', 'MonthlyCharges', 'TotalCharges', 'ViewingHoursPerWeek',
        'ContentDownloadsPerMonth', 'UserRating', 'SupportTicketsPerMonth', 'WatchlistSize'
    ]

    top10_categorical = [
        'SubscriptionType', 'DeviceRegistered'
    ]
    
    X = df[top10_features]
    y = df[target_col]
    
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    
    # 1) one-hot
    X_tr_oh, X_val_oh, X_test_oh = onehot_encode(X_train, X_val, X_test, top10_categorical)
    
    # One-hot w/o scaling
    save_dataset(X_tr_oh, X_val_oh, X_test_oh, y_train, y_val, y_test,
                base_path, feat_name, 'orig', 'onehot')
    
    # One-hot with scaling
    X_tr_scaled, X_val_scaled, X_test_scaled = scale_numeric(X_tr_oh, X_val_oh, X_test_oh, top10_numeric)
    save_dataset(X_tr_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test,
                base_path, feat_name, 'orig', 'onehot_scaled')

    # 2) Label encoding + category w/o scaling
    X_tr_le, X_val_le, X_test_le = label_encode_with_category_simple(X_train, X_val, X_test, top10_categorical)
    save_dataset(X_tr_le, X_val_le, X_test_le, y_train, y_val, y_test,
                base_path, feat_name, 'orig', 'label_category')

def save_dataset(X_tr, X_val, X_test, y_tr, y_val, y_test, base_path, feat_name, bal_key, type_name):

    path = os.path.join(base_path, 'datasets', f"{feat_name}_{bal_key}_{type_name}")
    os.makedirs(path, exist_ok=True)
    
    X_tr.to_csv(os.path.join(path, 'x_train.csv'), index=False)
    X_val.to_csv(os.path.join(path, 'x_val.csv'), index=False)
    X_test.to_csv(os.path.join(path, 'x_test.csv'), index=False)
    y_tr.to_csv(os.path.join(path, 'y_train.csv'), index=False)
    y_val.to_csv(os.path.join(path, 'y_val.csv'), index=False)
    y_test.to_csv(os.path.join(path, 'y_test.csv'), index=False)
    
    print(f"Saved: {feat_name}_{bal_key}_{type_name}")

def preprocessing():

    df, base_path = load_and_clean_data()
    
    os.makedirs(os.path.join(base_path, 'datasets'), exist_ok=True)
    
    create_all_features_datasets(df, base_path)
    
    create_top10_datasets(df, base_path)

    print(f"\n🎉 ALL DATASETS CREATED SUCCESSFULLY!")