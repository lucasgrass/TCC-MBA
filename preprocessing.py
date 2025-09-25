# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE

from utils import drop_outliers

# %%
def preprocessing():
    base_path = os.path.dirname(os.path.abspath(__file__))
    target_column = 'Churn'
    
    # --- Load dataset ---
    df = pd.read_csv(os.path.join(base_path, 'datasets/raw.csv'), sep=',', decimal='.')
    print("Churn value counts:\n", df['Churn'].value_counts())
    
    # --- Clean dataset ---
    print("\nNull values:", df.isnull().sum())
    df.dropna(inplace=True)
    
    print("\nDuplicated lines:", df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    
    # --- Remove outliers ---
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != target_column]
    print("\nBefore outliers removal:", df.shape)
    
    for col in numeric_cols:
        df = drop_outliers(df, col)
    
    print("After outliers removal:", df.shape)
    
    # --- Feature engineering ---

        # 1. COMPORTAMENTO DE USO E ENGAJAMENTO
    df['Avg_Session_Length_Minutes'] = df['ViewingHoursPerWeek'] * 60 / (df['AverageViewingDuration'] + 0.01)
    df['Content_Consumption_Score'] = df['ViewingHoursPerWeek'] * df['AverageViewingDuration']
    df['Downloads_per_ViewingHour'] = df['ContentDownloadsPerMonth'] / (df['ViewingHoursPerWeek'] + 0.01)
    
    # 2. RELAÇÃO CUSTO-VALOR (MUITO IMPORTANTE PARA CHURN)
    df['Value_Per_Hour'] = df['MonthlyCharges'] / (df['ViewingHoursPerWeek'] + 0.01)
    df['Charge_Increase_Ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] / (df['AccountAge'] + 0.01))
    df['Lifetime_Value_Score'] = df['TotalCharges'] / (df['AccountAge'] + 0.01)
    
    # 3. COMPORTAMENTO DE SUPORTE (CRÍTICO PARA CHURN)
    df['Support_Intensity'] = df['SupportTicketsPerMonth'] / (df['AccountAge'] + 0.01)
    df['Rating_vs_Support'] = df['UserRating'] - (df['SupportTicketsPerMonth'] * 0.5)  # quanto mais tickets, pior o rating relativo
    
    # 4. PADRÕES DE USUÁRIO
    df['Watchlist_Utilization'] = df['WatchlistSize'] / (df['ViewingHoursPerWeek'] + 0.01)
    df['Engagement_Consistency'] = df['ViewingHoursPerWeek'] / (df['AccountAge'] + 0.01)
    
    # 5. SINALIZADORES COMPORTAMENTAIS (MUITO PREDITIVOS)
    # Usuário caro que pouco usa
    df['High_Cost_Low_Usage'] = ((df['MonthlyCharges'] > df['MonthlyCharges'].median()) & 
                                (df['ViewingHoursPerWeek'] < df['ViewingHoursPerWeek'].median())).astype(int)
    
    # Muitos tickets em pouco tempo
    df['Support_Heavy_New_User'] = ((df['AccountAge'] < 6) & 
                                   (df['SupportTicketsPerMonth'] > 2)).astype(int)
    
    # Baixa avaliação apesar de bom uso
    df['Unhappy_Active_User'] = ((df['UserRating'] < 3) & 
                                (df['ViewingHoursPerWeek'] > df['ViewingHoursPerWeek'].median())).astype(int)
    
    # 6. INTERAÇÕES ENTRE VARIÁVEIS
    df['Premium_Low_Usage'] = ((df['SubscriptionType'] == 'Premium') & 
                              (df['ViewingHoursPerWeek'] < 10)).astype(int)
    
    df['High_Downloads_Low_Viewing'] = ((df['ContentDownloadsPerMonth'] > df['ContentDownloadsPerMonth'].median()) & 
                                       (df['ViewingHoursPerWeek'] < df['ViewingHoursPerWeek'].median())).astype(int)
    
        # Segmentação por valor do cliente
    df['Customer_Value_Segment'] = pd.cut(
        df['TotalCharges'], 
        bins=[0, 100, 500, 2000, df['TotalCharges'].max() + 1],
        labels=['low', 'medium', 'high', 'vip']
    )
    
    # Segmentação por engajamento
    df['Engagement_Segment'] = pd.cut(
        df['ViewingHoursPerWeek'], 
        bins=[0, 5, 15, 30, df['ViewingHoursPerWeek'].max() + 1],
        labels=['inactive', 'casual', 'active', 'power_user']
    )
    
    # Segmentação por suporte
    df['Support_Profile'] = pd.cut(
        df['SupportTicketsPerMonth'], 
        bins=[-1, 0, 2, 5, df['SupportTicketsPerMonth'].max() + 1],
        labels=['no_support', 'low_support', 'medium_support', 'high_support']
    )

    # df['AccountAge_Segment'] = pd.cut(
    #     df['AccountAge'], bins=[0, 24, 60, df['AccountAge'].max() + 1],
    #     labels=['new', 'standard', 'old']
    # )
    
    # df['SupportTicketsPerMonth_Segment'] = pd.cut(
    #     df['SupportTicketsPerMonth'], bins=[-1, 1, 5, df['SupportTicketsPerMonth'].max() + 1],
    #     labels=['low', 'medium', 'high']
    # )

    # df['Watchlist_per_Week'] = df['WatchlistSize'] / (df['ViewingHoursPerWeek'] + 0.01) 
    # df['AverageViewingDuration_per_Week'] = df['AverageViewingDuration'] / (df['ViewingHoursPerWeek'] + 0.01)

    # numeric_cols.extend(['Watchlist_per_Week', 'AverageViewingDuration_per_Week'])
    
    # Drop CustomerID
    if 'CustomerID' in df.columns:
        df.drop('CustomerID', axis=1, inplace=True)
    
    # --- Encoding ---
    print("Number of columns before encoding:", df.shape[1])
    
    label_encoder = LabelEncoder()
    df['SubscriptionType'] = label_encoder.fit_transform(df['SubscriptionType'])
    
    one_hot_encoder_variables = [
        'PaymentMethod', 'PaperlessBilling', 'ContentType', 'MultiDeviceAccess',
        'DeviceRegistered', 'GenrePreference', 'Gender', 'ParentalControl',
        'SubtitlesEnabled', 'AccountAge_Segment', 'SupportTicketsPerMonth_Segment'
    ]
    
    df = pd.get_dummies(df, columns=one_hot_encoder_variables, drop_first=False)
    
    print("Number of columns after encoding:", df.shape[1])
    
    # --- Split dataset ---
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, test_size=0.5, stratify=y_val_test, random_state=42
    )
    
    # --- StandardScaler ---
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train_scaled[numeric_cols])
    X_val_scaled[numeric_cols] = scaler.transform(X_val_scaled[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test_scaled[numeric_cols])
    
    # --- Undersampling (ENN) ---
    enn = EditedNearestNeighbours()
    X_train_under, y_train_under = enn.fit_resample(X_train_scaled, y_train)
    
    # --- Oversampling (SMOTE) ---
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_over, y_train_over = smote.fit_resample(X_train_scaled, y_train)
    
    # --- Save datasets ---
    datasets = [
        ('orig', X_train_scaled, y_train),
        ('under', X_train_under, y_train_under),
        ('over', X_train_over, y_train_over)
    ]
    
    for prefix, X_tr, y_tr in datasets:
        X_tr.to_csv(os.path.join(base_path, f'datasets/x_train_{prefix}.csv'), index=False)
        y_tr.to_csv(os.path.join(base_path, f'datasets/y_train_{prefix}.csv'), index=False)
        X_val_scaled.to_csv(os.path.join(base_path, f'datasets/x_val_{prefix}.csv'), index=False)
        y_val.to_csv(os.path.join(base_path, f'datasets/y_val_{prefix}.csv'), index=False)
        X_test_scaled.to_csv(os.path.join(base_path, f'datasets/x_test_{prefix}.csv'), index=False)
        y_test.to_csv(os.path.join(base_path, f'datasets/y_test_{prefix}.csv'), index=False)
    
    print(f"All datasets saved in: {base_path}")