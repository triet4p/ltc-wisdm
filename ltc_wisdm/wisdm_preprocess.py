import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# --- Constants Configuration ---
# This replaces the need for a separate config file for this script.

# 1. Path Configuration
# Assumes this script is in a directory like 'src/preprocessing/'
# and the project root is two levels up.
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / 'data'

# Specific paths for the WISDM dataset
WISDM_DATA_DIR = DATA_DIR / 'wisdm'
WISDM_RAW_DIR = WISDM_DATA_DIR / 'raw'
WISDM_PROCESSED_DIR = WISDM_DATA_DIR / 'processed'
RAW_DATA_FILE = WISDM_RAW_DIR / 'WISDM_ar_v1.1_raw.txt'

# 2. Data and Model Parameters
WINDOW_SIZE = 80
STEP_SIZE = 40
RAW_DATA_COL_NAMES = ['user', 'activity', 'timestamp', 'x', 'y', 'z']

def load_and_clean_data(path: Path) -> pd.DataFrame:
    """
    Load and clean raw WISDM data from the specified path.
    
    Args:
        path (Path): Path to the raw data file.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with sensor data.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at: {path}\n"
            f"Please make sure 'WISDM_ar_v1.1_raw.txt' is located in '{WISDM_RAW_DIR}'."
        )
        
    print(f"Loading raw data from: {path}")
    df = pd.read_csv(path,
                     header=None,
                     names=RAW_DATA_COL_NAMES,
                     on_bad_lines='skip')
    
    # Clean the 'z' column which may contain semicolons
    df['z'] = df['z'].astype(str).str.replace(';', '').astype(float)
    
    df.dropna(inplace=True)
    
    print(f"Data cleaned. Total rows: {len(df)}")
    print("Activity distribution:")
    print(df['activity'].value_counts())
    return df

def create_windows(df: pd.DataFrame, window_size: int, step_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows from the dataframe.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        window_size (int): The size of each window.
        step_size (int): The step size to move the window.
        
    Returns:
        A tuple containing the segments (X) and their corresponding labels (y).
    """
    print(f"Creating sliding windows (window_size={window_size}, step_size={step_size})...")
    segments = []
    labels = []
    
    # Group by user and activity to ensure windows don't mix data from different sources
    for (user, activity), group in df.groupby(['user', 'activity']):
        signal_data = group[['x', 'y', 'z']].values
        activity_labels = group['activity']
        
        for i in range(0, len(signal_data) - window_size, step_size):
            segment = signal_data[i : i + window_size]
            # The label for a window is the most frequent activity within it
            label = mode(activity_labels.iloc[i : i + window_size], keepdims=True)[0][0]
            
            segments.append(segment)
            labels.append(label)
            
    return np.array(segments), np.array(labels)

def wisdm_preprocess_pipeline():
    """
    Execute the complete preprocessing pipeline for the WISDM dataset.
    This includes loading, cleaning, windowing, encoding, standardizing, and saving data.
    """
    print("--- Starting WISDM Data Preprocessing Pipeline ---")
    
    # Ensure the target directory exists
    os.makedirs(WISDM_PROCESSED_DIR, exist_ok=True)
    
    # 1. Load and Clean Data
    df = load_and_clean_data(RAW_DATA_FILE)
    
    # 2. Create Sliding Windows
    X, y_str = create_windows(df, WINDOW_SIZE, STEP_SIZE)
    print(f"Successfully created {len(X)} windows.")
    print(f"Shape of feature matrix X: {X.shape}")
    
    # 3. Label Encoding
    print("Encoding string labels to integers...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)
    print("Classes and their corresponding integer encodings:")
    print(f"{list(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # 4. Split into Training and Test sets
    print("Splitting data into training and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set size: X={X_test.shape}, y={y_test.shape}")
    
    # 5. Standard Scaling
    print("Standardizing data (fitting scaler on training data only)...")
    # Reshape for scaler: (num_samples * num_timesteps, num_features)
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    
    scaler = StandardScaler()
    # Fit the scaler ONLY on the training data to prevent data leakage
    scaler.fit(X_train_reshaped)
    
    # Apply the scaler to both train and test sets
    X_train = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    X_test = scaler.transform(X_test_reshaped).reshape(X_test.shape)
    
    print("Standardization completed.")
    
    # 6. Save Processed Data and Artifacts
    print(f"Saving processed data to directory: {WISDM_PROCESSED_DIR}")
    
    np.save(WISDM_PROCESSED_DIR / 'X_train.npy', X_train)
    np.save(WISDM_PROCESSED_DIR / 'y_train.npy', y_train)
    np.save(WISDM_PROCESSED_DIR / 'X_test.npy', X_test)
    np.save(WISDM_PROCESSED_DIR / 'y_test.npy', y_test)
    
    joblib.dump(scaler, WISDM_PROCESSED_DIR / 'scaler.pkl')
    joblib.dump(label_encoder, WISDM_PROCESSED_DIR / 'label_encoder.pkl')
    
    print("--- WISDM Data Preprocessing Completed Successfully! ---")

if __name__ == '__main__':
    # This allows the script to be run directly from the command line
    wisdm_preprocess_pipeline()