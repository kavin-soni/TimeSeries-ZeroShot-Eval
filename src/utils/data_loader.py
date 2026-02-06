import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

def load_and_prepare_data(dataset_name, train_file, test_file, config):
    """
    Generic loader for the datasets used in the paper.
    """
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"Error: {train_file} or {test_file} not found.")
        return None

    df_train = pd.read_csv(train_file, low_memory=False)
    df_test = pd.read_csv(test_file, low_memory=False)

    # Standardize column names
    for df in [df_train, df_test]:
        df['series_id'] = df['series_id'].astype(str)
        if 'value' in df.columns:
            df.rename(columns={'value': 'sales'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])

    if dataset_name in ['traffic', 'etth1', 'exchange']:
        return prepare_supervised_data(df_train, df_test, config)
    else:
        # M4 simple return
        return df_train, df_test

def prepare_supervised_data(df_train, df_test, config):
    """
    Generates features for supervised baselines.
    """
    lag_shift = config.get('lag_shift', 1)
    
    df_train['is_test'] = False
    df_test['is_test'] = True
    
    full_df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    full_df = full_df.sort_values(['series_id', 'date'])
    
    # Feature Engineering
    full_df['year'] = full_df['date'].dt.year
    full_df['month'] = full_df['date'].dt.month
    full_df['day'] = full_df['date'].dt.day
    full_df['weekday'] = full_df['date'].dt.weekday
    if 'hour' in full_df['date'].dt.__dir__():
        full_df['hour'] = full_df['date'].dt.hour
    
    encoder = LabelEncoder()
    full_df['series_id_encoded'] = encoder.fit_transform(full_df['series_id'])
    
    # Lags
    for lag in config.get('lags', [1]):
        full_df[f'lag_{lag}'] = full_df.groupby('series_id')['sales'].shift(lag_shift + lag - 1)
        
    # Rolling stats
    for window in config.get('windows', [24]):
        full_df[f'rolling_mean_{window}'] = full_df.groupby('series_id')['sales'].transform(
            lambda x: x.shift(lag_shift).rolling(window).mean()
        )

    # Split back
    X_test_full = full_df[full_df['is_test'] == True].copy()
    train_full = full_df[full_df['is_test'] == False].copy()
    
    drop_cols = ['sales', 'date', 'series_id', 'is_test', 'split_type']
    features = [c for c in full_df.columns if c not in drop_cols]
    
    return train_full, X_test_full, features
