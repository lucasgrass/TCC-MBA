# %%
import os
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from utils import drop_outliers, onehot_encode, label_encode_with_category_simple, scale_numeric, train_val_test_split

def load_and_clean_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base_path, 'datasets/raw.csv'), sep=',', decimal='.')
    
    df.drop('CustomerID', axis=1, inplace=True)
    df.dropna(inplace=True)
    df = df.drop_duplicates().reset_index(drop=True)

    numeric_cols = [
      'AccountAge', 'MonthlyCharges', 'TotalCharges',
      'ViewingHoursPerWeek', 'AverageViewingDuration',
      'ContentDownloadsPerMonth', 'UserRating',
      'SupportTicketsPerMonth', 'WatchlistSize'
    ]

    print('\nBefore outliers function:', df.shape)
    for col in numeric_cols:
        df = drop_outliers(df, col)
    print('\nAfter outliers function:', df.shape)

    print(f"Number of columns: {len(df.columns)}")
    
    return df, base_path

def balance_train(X_train, y_train):
    rus = RandomUnderSampler(random_state=42)
    smote = BorderlineSMOTE(random_state=42, kind='borderline-1')
    
    X_under, y_under = rus.fit_resample(X_train, y_train)
    X_over, y_over = smote.fit_resample(X_train, y_train)
    
    return {
        'orig': (X_train, y_train),
        'under': (X_under, y_under),
        'over': (X_over, y_over)
    }

def create_all_features_datasets(df, base_path):
    target_col = 'Churn'
    feat_name = 'all'

    numeric_cols = [
        'AccountAge', 'MonthlyCharges', 'TotalCharges',
        'ViewingHoursPerWeek', 'AverageViewingDuration',
        'ContentDownloadsPerMonth', 'UserRating',
        'SupportTicketsPerMonth', 'WatchlistSize'
    ]

    categorical_cols = [
        'SubscriptionType', 'PaymentMethod', 'PaperlessBilling',
        'ContentType', 'MultiDeviceAccess', 'DeviceRegistered',
        'GenrePreference', 'Gender', 'ParentalControl', 'SubtitlesEnabled'
    ]

    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    # -------------------------
    # 1) One-hot only
    # -------------------------
    X_tr_oh, X_val_oh, X_test_oh = onehot_encode(X_train, X_val, X_test, categorical_cols)
    train_dict = balance_train(X_tr_oh, y_train)
    save_balanced_datasets(
        {k: v[0] for k, v in train_dict.items()},
        X_val_oh, X_test_oh,
        {k: v[1] for k, v in train_dict.items()},
        y_val, y_test,
        base_path, feat_name, "onehot"
    )

    # -------------------------
    # 2) Label encoding + category
    # -------------------------
    X_tr_le, X_val_le, X_test_le = label_encode_with_category_simple(X_train, X_val, X_test, categorical_cols)
    train_dict = balance_train(X_tr_le, y_train)
    save_balanced_datasets(
        {k: v[0] for k, v in train_dict.items()},
        X_val_le, X_test_le,
        {k: v[1] for k, v in train_dict.items()},
        y_val, y_test,
        base_path, feat_name, "label_category"
    )

    # -------------------------
    # 3) One-hot + scaling
    # -------------------------
    X_tr_oh2, X_val_oh2, X_test_oh2 = onehot_encode(X_train, X_val, X_test, categorical_cols)
    X_tr_scaled, X_val_scaled, X_test_scaled = scale_numeric(X_tr_oh2, X_val_oh2, X_test_oh2, numeric_cols)
    train_dict = balance_train(X_tr_scaled, y_train)

    save_balanced_datasets(
        {k: v[0] for k, v in train_dict.items()},
        X_val_scaled, X_test_scaled,
        {k: v[1] for k, v in train_dict.items()},
        y_val, y_test,
        base_path, feat_name, "onehot_scaled"
    )

def create_top10_datasets(df, base_path):
    target_col = 'Churn'
    feat_name = 'top10'
    
    top10_features = [
        'AccountAge', 'MonthlyCharges', 'TotalCharges', 'SubscriptionType',
        'DeviceRegistered', 'ViewingHoursPerWeek', 'ContentDownloadsPerMonth',
        'UserRating', 'SupportTicketsPerMonth', 'WatchlistSize'
    ]
    
    numeric_cols = [
        'AccountAge', 'MonthlyCharges', 'TotalCharges', 'ViewingHoursPerWeek',
        'ContentDownloadsPerMonth', 'UserRating', 'SupportTicketsPerMonth',
        'WatchlistSize'
    ]
    
    categorical_cols = ['SubscriptionType', 'DeviceRegistered']
    
    X = df[top10_features]
    y = df[target_col]
    
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    # -------------------------
    # 1) One-hot only
    # -------------------------
    X_tr_oh, X_val_oh, X_test_oh = onehot_encode(X_train, X_val, X_test, categorical_cols)
    train_dict = balance_train(X_tr_oh, y_train)
    save_balanced_datasets(
        {k: v[0] for k, v in train_dict.items()},
        X_val_oh, X_test_oh,
        {k: v[1] for k, v in train_dict.items()},
        y_val, y_test,
        base_path, feat_name, "onehot"
    )

    # -------------------------
    # 2) Label encoding + category
    # -------------------------
    X_tr_le, X_val_le, X_test_le = label_encode_with_category_simple(X_train, X_val, X_test, categorical_cols)
    train_dict = balance_train(X_tr_le, y_train)
    save_balanced_datasets(
        {k: v[0] for k, v in train_dict.items()},
        X_val_le, X_test_le,
        {k: v[1] for k, v in train_dict.items()},
        y_val, y_test,
        base_path, feat_name, "label_category"
    )

    # -------------------------
    # 3) One-hot + scaling
    # -------------------------
    X_tr_oh2, X_val_oh2, X_test_oh2 = onehot_encode(X_train, X_val, X_test, categorical_cols)
    X_tr_scaled, X_val_scaled, X_test_scaled = scale_numeric(X_tr_oh2, X_val_oh2, X_test_oh2, numeric_cols)
    train_dict = balance_train(X_tr_scaled, y_train)
    save_balanced_datasets(
        {k: v[0] for k, v in train_dict.items()},
        X_val_scaled, X_test_scaled,
        {k: v[1] for k, v in train_dict.items()},
        y_val, y_test,
        base_path, feat_name, "onehot_scaled"
    )

def create_top10_fe_datasets(df, base_path):
    target_col = 'Churn'
    feat_name = 'top10_fe'
    
    top10_features = [
        'AccountAge', 'MonthlyCharges', 'TotalCharges', 'SubscriptionType',
        'DeviceRegistered', 'ViewingHoursPerWeek', 'ContentDownloadsPerMonth',
        'UserRating', 'SupportTicketsPerMonth', 'WatchlistSize'
    ]
    
    df = df.copy()

    df["ValuePerHour"] = df["MonthlyCharges"] / (df["ViewingHoursPerWeek"] * 4 + 0.1)
    df['High_Cost_Low_Usage'] = (
        (df['MonthlyCharges'] > df['MonthlyCharges'].median()) &
        (df['ViewingHoursPerWeek'] < df['ViewingHoursPerWeek'].median())
    ).astype(int)
    df['Support_Intensity'] = df['SupportTicketsPerMonth'] / (df['AccountAge'] + 0.1)
    
    used_features = top10_features + ['ValuePerHour', 'High_Cost_Low_Usage', 'Support_Intensity']
    
    numeric_cols = [
        'AccountAge', 'MonthlyCharges', 'TotalCharges', 'ViewingHoursPerWeek',
        'ContentDownloadsPerMonth', 'UserRating', 'SupportTicketsPerMonth',
        'WatchlistSize', 'ValuePerHour', 'Support_Intensity'
    ]
    
    categorical_cols = ['SubscriptionType', 'DeviceRegistered', 'High_Cost_Low_Usage']
    
    X = df[used_features]
    y = df[target_col]
    
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    # -------------------------
    # 1) One-hot only
    # -------------------------
    X_tr_oh, X_val_oh, X_test_oh = onehot_encode(X_train, X_val, X_test, categorical_cols)
    train_dict = balance_train(X_tr_oh, y_train)
    save_balanced_datasets(
        {k: v[0] for k, v in train_dict.items()},
        X_val_oh, X_test_oh,
        {k: v[1] for k, v in train_dict.items()},
        y_val, y_test,
        base_path, feat_name, "onehot"
    )

    # -------------------------
    # 2) Label encoding + category
    # -------------------------
    X_tr_le, X_val_le, X_test_le = label_encode_with_category_simple(X_train, X_val, X_test, categorical_cols)
    train_dict = balance_train(X_tr_le, y_train)
    save_balanced_datasets(
        {k: v[0] for k, v in train_dict.items()},
        X_val_le, X_test_le,
        {k: v[1] for k, v in train_dict.items()},
        y_val, y_test,
        base_path, feat_name, "label_category"
    )

    # -------------------------
    # 3) One-hot + scaling
    # -------------------------
    X_tr_oh2, X_val_oh2, X_test_oh2 = onehot_encode(X_train, X_val, X_test, categorical_cols)
    X_tr_scaled, X_val_scaled, X_test_scaled = scale_numeric(X_tr_oh2, X_val_oh2, X_test_oh2, numeric_cols)
    train_dict = balance_train(X_tr_scaled, y_train)
    save_balanced_datasets(
        {k: v[0] for k, v in train_dict.items()},
        X_val_scaled, X_test_scaled,
        {k: v[1] for k, v in train_dict.items()},
        y_val, y_test,
        base_path, feat_name, "onehot_scaled"
    )

def save_balanced_datasets(X_train_dict, X_val, X_test, y_train_dict, y_val, y_test, base_path, feat_name, type_prefix):
    for bal_key in X_train_dict.keys():
        save_dataset(
            X_train_dict[bal_key], X_val, X_test,
            y_train_dict[bal_key], y_val, y_test,
            base_path, feat_name, f"{type_prefix}_{bal_key}"
        )

def save_dataset(X_tr, X_val, X_test, y_tr, y_val, y_test, base_path, feat_name, type_dataset):
    path = os.path.join(base_path, 'datasets', f"{feat_name}_{type_dataset}")
    os.makedirs(path, exist_ok=True)
    
    X_tr.to_csv(os.path.join(path, 'x_train.csv'), index=False)
    X_val.to_csv(os.path.join(path, 'x_val.csv'), index=False)
    X_test.to_csv(os.path.join(path, 'x_test.csv'), index=False)
    y_tr.to_csv(os.path.join(path, 'y_train.csv'), index=False)
    y_val.to_csv(os.path.join(path, 'y_val.csv'), index=False)
    y_test.to_csv(os.path.join(path, 'y_test.csv'), index=False)
    
    print(f"Saved: {feat_name}_{type_dataset}")


def preprocessing():
    df, base_path = load_and_clean_data()
    os.makedirs(os.path.join(base_path, 'datasets'), exist_ok=True)

    create_all_features_datasets(df, base_path)
    create_top10_datasets(df, base_path)
    create_top10_fe_datasets(df, base_path)

    print(f"\nALL DATASETS CREATED SUCCESSFULLY!")
