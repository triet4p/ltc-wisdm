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
    df = pd.read_csv(path,
                     header=None,
                     names=cfg.RAW_DATA_COL_NAMES,
                     on_bad_lines='skip')
    
    df['z'] = df['z'].astype(str).str.replace(';', '').astype(float)
    
    df.dropna(inplace=True)
    
    print(f"Dữ liệu đã được làm sạch. Tổng số dòng: {len(df)}")
    print("Phân phối các hoạt động:")
    print(df['activity'].value_counts())
    return df

def create_windows(df: pd.DataFrame, window_size: int, step_size: int):
    """Tạo các cửa sổ trượt từ dataframe."""
    print(f"Đang tạo cửa sổ trượt (window_size={window_size}, step_size={step_size})...")
    segments = []
    labels = []
    
    # Group theo user và activity để đảm bảo cửa sổ không bị lẫn lộn dữ liệu
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
    os.makedirs(cfg.PROCESSED_DATA_DIR, exist_ok=True)
    
    df = load_and_clean_data(cfg.RAW_DATA_FILE)
    X, y_str = create_windows(df, cfg.WINDOW_SIZE, cfg.STEP_SIZE)
    print(f"Tạo thành công {len(X)} cửa sổ.")
    print(f"Shape của X: {X.shape}") # Sẽ là (số cửa sổ, 80, 3)
    
    # 4. Mã hóa nhãn (Label Encoding)
    print("Đang mã hóa nhãn...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)
    print(f"Các lớp và mã hóa tương ứng: {list(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    print("Đang chia dữ liệu train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Kích thước tập huấn luyện: {X_train.shape}, {y_train.shape}")
    print(f"Kích thước tập kiểm tra: {X_test.shape}, {y_test.shape}")
    
    # 6. Chuẩn hóa dữ liệu (Standard Scaling)
    print("Đang chuẩn hóa dữ liệu...")
    # Reshape để scaler xử lý được: (samples * timesteps, features)
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    
    scaler = StandardScaler()
    # Fit scaler CHỈ trên dữ liệu train để tránh rò rỉ dữ liệu
    scaler.fit(X_train_reshaped)
    
    # Áp dụng scaler cho cả train và test
    X_train = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    
    # Reshape và transform cho tập test
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    X_test = scaler.transform(X_test_reshaped).reshape(X_test.shape)
    
    print("Chuẩn hóa hoàn tất.")
    
    # 7. Lưu kết quả
    print(f"Đang lưu dữ liệu đã xử lý vào thư mục: {cfg.PROCESSED_DATA_DIR}")
    os.makedirs(cfg.PROCESSED_DATA_DIR, exist_ok=True)
    
    np.save(os.path.join(cfg.PROCESSED_DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(cfg.PROCESSED_DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(cfg.PROCESSED_DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(cfg.PROCESSED_DATA_DIR, 'y_test.npy'), y_test)
    
    # Lưu cả scaler và label_encoder để dùng lại sau này
    joblib.dump(scaler, os.path.join(cfg.PROCESSED_DATA_DIR, 'scaler.pkl'))
    joblib.dump(label_encoder, os.path.join(cfg.PROCESSED_DATA_DIR, 'label_encoder.pkl'))
    
    print("--- Hoàn thành tiền xử lý dữ liệu! ---")