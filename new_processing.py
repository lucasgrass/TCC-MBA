# preprocessing.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE

def drop_outliers(df, column):
    """Remove outliers usando IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def new_preprocessing():
    base_path = os.path.dirname(os.path.abspath(__file__))
    target_column = 'Churn'
    
    # --- Load dataset ---
    df = pd.read_csv(os.path.join(base_path, 'datasets/raw.csv'), sep=',', decimal='.')
    print("Dataset shape:", df.shape)
    print("Churn value counts:\n", df['Churn'].value_counts())
    print("Churn rate:", f"{df['Churn'].mean():.2%}")
    
    # --- Clean dataset ---
    print("\n=== DATA CLEANING ===")
    print("Null values:", df.isnull().sum().sum())
    if df.isnull().sum().sum() > 0:
        print("Detailed null values:")
        print(df.isnull().sum())
    df.dropna(inplace=True)
    
    print("Duplicated lines:", df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    print("Shape after cleaning:", df.shape)
    
    # --- Remove outliers apenas de colunas numéricas específicas ---
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in [target_column, 'CustomerID'] and df[col].nunique() > 10]
    
    print("\n=== OUTLIER REMOVAL ===")
    print("Numeric columns for outlier removal:", numeric_cols)
    print("Before outlier removal:", df.shape)
    
    for col in numeric_cols:
        initial_shape = df.shape[0]
        df = drop_outliers(df, col)
        final_shape = df.shape[0]
        if initial_shape != final_shape:
            print(f"Removed {initial_shape - final_shape} outliers from {col}")
    
    print("After outlier removal:", df.shape)
    
    # --- FEATURE ENGINEERING MELHORADO ---
    print("\n=== FEATURE ENGINEERING ===")
    
    # 1. COMPORTAMENTO DE USO E ENGAJAMENTO
    df['Avg_Session_Length_Minutes'] = df['ViewingHoursPerWeek'] * 60 / (df['AverageViewingDuration'] + 0.01)
    df['Content_Consumption_Score'] = df['ViewingHoursPerWeek'] * df['AverageViewingDuration']
    df['Downloads_per_ViewingHour'] = df['ContentDownloadsPerMonth'] / (df['ViewingHoursPerWeek'] + 0.01)
    df['Efficiency_Score'] = df['ContentDownloadsPerMonth'] / (df['ViewingHoursPerWeek'] + 0.01)
    
    # 2. RELAÇÃO CUSTO-VALOR (MUITO IMPORTANTE PARA CHURN)
    df['Value_Per_Hour'] = df['MonthlyCharges'] / (df['ViewingHoursPerWeek'] + 0.01)
    df['Charge_Increase_Ratio'] = df['MonthlyCharges'] / ((df['TotalCharges'] / (df['AccountAge'] + 0.01)) + 0.01)
    df['Lifetime_Value_Score'] = df['TotalCharges'] / (df['AccountAge'] + 0.01)
    df['Monthly_Value_Ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 0.01)
    
    # 3. COMPORTAMENTO DE SUPORTE (CRÍTICO PARA CHURN)
    df['Support_Intensity'] = df['SupportTicketsPerMonth'] / (df['AccountAge'] + 0.01)
    df['Rating_vs_Support'] = df['UserRating'] - (df['SupportTicketsPerMonth'] * 0.5)
    df['Support_per_ViewingHour'] = df['SupportTicketsPerMonth'] / (df['ViewingHoursPerWeek'] + 0.01)
    
    # 4. PADRÕES DE USUÁRIO
    df['Watchlist_Utilization'] = df['WatchlistSize'] / (df['ViewingHoursPerWeek'] + 0.01)
    df['Engagement_Consistency'] = df['ViewingHoursPerWeek'] / (df['AccountAge'] + 0.01)
    df['Account_Maturity'] = np.log(df['AccountAge'] + 1)
    
    # 5. SINALIZADORES COMPORTAMENTAIS (MUITO PREDITIVOS)
    df['High_Cost_Low_Usage'] = ((df['MonthlyCharges'] > df['MonthlyCharges'].median()) & 
                                (df['ViewingHoursPerWeek'] < df['ViewingHoursPerWeek'].median())).astype(int)
    
    df['Support_Heavy_New_User'] = ((df['AccountAge'] < 6) & 
                                   (df['SupportTicketsPerMonth'] > 2)).astype(int)
    
    df['Unhappy_Active_User'] = ((df['UserRating'] < 3) & 
                                (df['ViewingHoursPerWeek'] > df['ViewingHoursPerWeek'].median())).astype(int)
    
    df['Premium_Low_Usage'] = ((df['SubscriptionType'] == 'Premium') & 
                              (df['ViewingHoursPerWeek'] < 10)).astype(int)
    
    df['High_Downloads_Low_Viewing'] = ((df['ContentDownloadsPerMonth'] > df['ContentDownloadsPerMonth'].median()) & 
                                       (df['ViewingHoursPerWeek'] < df['ViewingHoursPerWeek'].median())).astype(int)
    
    df['Low_Rating_High_Cost'] = ((df['UserRating'] < 2.5) & 
                                 (df['MonthlyCharges'] > df['MonthlyCharges'].median())).astype(int)
    
    print(f"Created {len([col for col in df.columns if col not in ['CustomerID', 'Churn']])} features total")

    # --- SEGMENTAÇÃO MELHORADA ---
    print("\n=== SEGMENTATION ===")
    
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
    
    # Segmentação por idade da conta
    df['Account_Age_Segment'] = pd.cut(
        df['AccountAge'],
        bins=[0, 6, 12, 24, 60, df['AccountAge'].max() + 1],
        labels=['new', 'young', 'established', 'mature', 'veteran']
    )

    # --- ENCODING OTIMIZADO ---
    print("\n=== ENCODING ===")
    
    # Save original for analysis
    original_columns = df.columns.tolist()
    
    # Para SubscriptionType, ordene pelo valor
    subscription_map = {'Basic': 0, 'Standard': 1, 'Premium': 2}
    df['SubscriptionType_encoded'] = df['SubscriptionType'].map(subscription_map)
    
    # Target encoding para categorias importantes
    categorical_for_target_encoding = ['DeviceRegistered', 'GenrePreference', 'PaymentMethod', 'ContentType']
    
    for col in categorical_for_target_encoding:
        if col in df.columns:
            churn_rate_by_cat = df.groupby(col)['Churn'].mean()
            df[f'{col}_ChurnRate'] = df[col].map(churn_rate_by_cat)
            print(f"Target encoded {col} - unique values: {churn_rate_by_cat.to_dict()}")
    
    # One-hot para as demais categorias
    one_hot_encoder_variables = [
        'PaymentMethod', 'PaperlessBilling', 'ContentType', 'MultiDeviceAccess',
        'DeviceRegistered', 'GenrePreference', 'Gender', 'ParentalControl',
        'SubtitlesEnabled', 'Customer_Value_Segment', 'Engagement_Segment', 
        'Support_Profile', 'Account_Age_Segment'
    ]
    
    # Filtrar colunas que existem no dataframe
    one_hot_encoder_variables = [col for col in one_hot_encoder_variables if col in df.columns]
    
    print(f"One-hot encoding {len(one_hot_encoder_variables)} variables")
    df = pd.get_dummies(df, columns=one_hot_encoder_variables, drop_first=True)
    
    # --- ANÁLISE EXPLORATÓRIA ---
    print("\n=== EXPLORATORY ANALYSIS ===")
    
    # Novas features numéricas para análise
    new_numeric_features = [
        'Avg_Session_Length_Minutes', 'Content_Consumption_Score', 'Downloads_per_ViewingHour',
        'Value_Per_Hour', 'Charge_Increase_Ratio', 'Lifetime_Value_Score', 'Monthly_Value_Ratio',
        'Support_Intensity', 'Rating_vs_Support', 'Support_per_ViewingHour', 'Watchlist_Utilization',
        'Engagement_Consistency', 'Account_Maturity', 'SubscriptionType_encoded'
    ]
    
    # Adicionar target encoded features
    target_encoded_features = [f'{col}_ChurnRate' for col in categorical_for_target_encoding]
    new_numeric_features.extend(target_encoded_features)
    
    # Adicionar flags comportamentais
    behavioral_flags = ['High_Cost_Low_Usage', 'Support_Heavy_New_User', 'Unhappy_Active_User',
                       'Premium_Low_Usage', 'High_Downloads_Low_Viewing', 'Low_Rating_High_Cost']
    new_numeric_features.extend(behavioral_flags)
    
    # Filtrar colunas que existem
    new_numeric_features = [col for col in new_numeric_features if col in df.columns]
    
    # Calcular correlações
    analysis_cols = new_numeric_features + ['Churn']
    correlation_with_churn = df[analysis_cols].corr()['Churn'].sort_values(ascending=False)
    
    print("\n🔍 TOP 15 CORRELAÇÕES COM CHURN:")
    for feature, corr in correlation_with_churn.head(15).items():
        if feature != 'Churn':
            print(f"  {feature}: {corr:+.3f}")
    
    print("\n📉 BOTTOM 5 CORRELAÇÕES COM CHURN:")
    for feature, corr in correlation_with_churn.tail(6).items():
        if feature != 'Churn':
            print(f"  {feature}: {corr:+.3f}")
    
    # Plot das top correlações
    top_corr_features = correlation_with_churn.drop('Churn').head(10)
    plt.figure(figsize=(10, 6))
    top_corr_features.plot(kind='barh')
    plt.title('Top 10 Features com Maior Correlação com Churn')
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, 'correlation_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- PREPARE FOR MODELING ---
    print("\n=== DATA PREPARATION ===")
    
    # Drop CustomerID e colunas originais redundantes
    columns_to_drop = ['CustomerID']
    for col in ['SubscriptionType', 'DeviceRegistered', 'GenrePreference', 'PaymentMethod', 'ContentType']:
        if col in df.columns:
            columns_to_drop.append(col)
    
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
    # Definir X e y
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print(f"Final dataset shape: {X.shape}")
    print(f"Churn rate: {y.mean():.2%}")
    
    # --- Split dataset ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # --- StandardScaler ---
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    print(f"After scaling - X_train: {X_train_scaled.shape}, X_test: {X_test_scaled.shape}")
    
    # --- Undersampling (ENN) ---
    enn = EditedNearestNeighbours()
    X_train_under, y_train_under = enn.fit_resample(X_train_scaled, y_train)
    
    # --- Oversampling (SMOTE) ---
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_over, y_train_over = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"After sampling - Under: {X_train_under.shape}, Over: {X_train_over.shape}")
    
    # --- Save datasets ---
    datasets = [
        ('orig', X_train_scaled, y_train, X_test_scaled, y_test),
        ('under', X_train_under, y_train_under, X_test_scaled, y_test),
        ('over', X_train_over, y_train_over, X_test_scaled, y_test)
    ]
    
    for prefix, X_tr, y_tr, X_te, y_te in datasets:
        X_tr.to_csv(os.path.join(base_path, f'datasets/x_train_{prefix}.csv'), index=False)
        y_tr.to_csv(os.path.join(base_path, f'datasets/y_train_{prefix}.csv'), index=False)
        X_te.to_csv(os.path.join(base_path, f'datasets/x_test_{prefix}.csv'), index=False)
        y_te.to_csv(os.path.join(base_path, f'datasets/y_test_{prefix}.csv'), index=False)
    
    # Save feature names
    feature_names = X.columns.tolist()
    pd.DataFrame(feature_names, columns=['feature_name']).to_csv(
        os.path.join(base_path, 'datasets/feature_names.csv'), index=False
    )
    
    print(f"\n✅ All datasets saved in: {base_path}/datasets/")
    print(f"✅ Feature analysis saved as: correlation_analysis.png")
    print(f"✅ Total features: {len(feature_names)}")
    print(f"✅ Final churn rate: {y.mean():.2%}")