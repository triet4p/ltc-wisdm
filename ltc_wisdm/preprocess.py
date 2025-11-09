import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

from . import config as cfg

def load_and_clean_data(path: Path):
    """
    Load and clean raw data from the specified path.
    
    Args:
        path (Path): Path to the raw data file.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with sensor data.
    """
    df = pd.read_csv(path,
                     header=None,
                     names=cfg.RAW_DATA_COL_NAMES,
                     on_bad_lines='skip')
    
    df['z'] = df['z'].astype(str).str.replace(';', '').astype(float)
    
    df.dropna(inplace=True)
    
    print(f"Data cleaned. Total rows: {len(df)}")
    print("Activity distribution:")
    print(df['activity'].value_counts())
    return df

def create_windows(df: pd.DataFrame, window_size: int, step_size: int):
    """Create sliding windows from the dataframe."""
    print(f"Creating sliding windows (window_size={window_size}, step_size={step_size})...")
    segments = []
    labels = []
    
    # Group by user and activity to ensure windows don't mix data
    for (user, activity), group in df.groupby(['user', 'activity']):
        signal_data = group[['x', 'y', 'z']].values
        activity_labels = group['activity']
        
        for i in range(0, len(signal_data) - window_size, step_size):
            segment = signal_data[i : i + window_size]
            label = activity_labels.iloc[i : i + window_size].mode()[0]
            
            segments.append(segment)
            labels.append(label)
            
    return np.array(segments), np.array(labels)

def preprocess_pipeline():
    """
    Execute the complete preprocessing pipeline for WISDM dataset.
    This includes loading, cleaning, windowing, encoding, and standardizing data.
    """
    os.makedirs(cfg.PROCESSED_DATA_DIR, exist_ok=True)
    
    df = load_and_clean_data(cfg.RAW_DATA_FILE)
    X, y_str = create_windows(df, cfg.WINDOW_SIZE, cfg.STEP_SIZE)
    print(f"Successfully created {len(X)} windows.")
    print(f"Shape of X: {X.shape}") # Will be (number of windows, 80, 3)
    
    # 4. Label Encoding
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)
    print(f"Classes and corresponding encodings: {list(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    print("Splitting train/test data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape}, {y_train.shape}")
    print(f"Test set size: {X_test.shape}, {y_test.shape}")
    
    # 6. Standard Scaling
    print("Standardizing data...")
    # Reshape so scaler can handle: (samples * timesteps, features)
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    
    scaler = StandardScaler()
    # Fit scaler ONLY on training data to prevent data leakage
    scaler.fit(X_train_reshaped)
    
    # Apply scaler to both train and test
    X_train = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    
    # Reshape and transform for test set
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    X_test = scaler.transform(X_test_reshaped).reshape(X_test.shape)
    
    print("Standardization completed.")
    
    # 7. Save results
    print(f"Saving processed data to directory: {cfg.PROCESSED_DATA_DIR}")
    os.makedirs(cfg.PROCESSED_DATA_DIR, exist_ok=True)
    
    np.save(os.path.join(cfg.PROCESSED_DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(cfg.PROCESSED_DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(cfg.PROCESSED_DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(cfg.PROCESSED_DATA_DIR, 'y_test.npy'), y_test)
    
    # Save scaler and label_encoder for later use
    joblib.dump(scaler, os.path.join(cfg.PROCESSED_DATA_DIR, 'scaler.pkl'))
    joblib.dump(label_encoder, os.path.join(cfg.PROCESSED_DATA_DIR, 'label_encoder.pkl'))
    
    print("--- Data preprocessing completed! ---")