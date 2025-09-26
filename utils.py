import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def drop_outliers(df, column):
    
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def onehot_encode(X_train, X_val, X_test, categorical_cols):
    X_train_enc = pd.get_dummies(X_train, columns=categorical_cols, drop_first=False)
    X_val_enc = pd.get_dummies(X_val, columns=categorical_cols, drop_first=False)
    X_test_enc = pd.get_dummies(X_test, columns=categorical_cols, drop_first=False)
    
    all_columns = set(X_train_enc.columns) | set(X_val_enc.columns) | set(X_test_enc.columns)
    for dataset in [X_train_enc, X_val_enc, X_test_enc]:
        for col in all_columns:
            if col not in dataset.columns:
                dataset[col] = 0
        dataset = dataset[list(all_columns)]
    
    return X_train_enc, X_val_enc, X_test_enc

def label_encode(X_train, X_val, X_test, categorical_cols):
    X_train_enc = X_train.copy()
    X_val_enc = X_val.copy()
    X_test_enc = X_test.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train[col], X_val[col], X_test[col]], axis=0)
        le.fit(combined)
        X_train_enc[col] = le.transform(X_train[col])
        X_val_enc[col] = le.transform(X_val[col])
        X_test_enc[col] = le.transform(X_test[col])
    
    return X_train_enc, X_val_enc, X_test_enc

def label_encode_with_category_simple(X_train, X_val, X_test, categorical_cols):

    X_train_enc = X_train.copy()
    X_val_enc = X_val.copy()
    X_test_enc = X_test.copy()
    
    for col in categorical_cols:
        combined = pd.concat([X_train[col], X_val[col], X_test[col]], axis=0)
        
        le = LabelEncoder()
        le.fit(combined)
        
        X_train_enc[col] = le.transform(X_train[col])
        X_val_enc[col] = le.transform(X_val[col])
        X_test_enc[col] = le.transform(X_test[col])
        
        X_train_enc[col] = X_train_enc[col].astype('category')
        X_val_enc[col] = X_val_enc[col].astype('category')
        X_test_enc[col] = X_test_enc[col].astype('category')
    
    return X_train_enc, X_val_enc, X_test_enc

def scale_numeric(X_train, X_val, X_test, numeric_cols):
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train_scaled[numeric_cols])
    X_val_scaled[numeric_cols] = scaler.transform(X_val_scaled[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test_scaled[numeric_cols])
    
    return X_train_scaled, X_val_scaled, X_test_scaled

def train_val_test_split(X, y, test_size=0.2, val_size=0.25, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, random_state=random_state
    )
    return X_train_final, X_val, X_test, y_train_final, y_val, y_test