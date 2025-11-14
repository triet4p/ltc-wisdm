import os
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

# --- Constants Configuration ---

# 1. Path Configuration
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / 'data'
PENDULUM_DATA_DIR = DATA_DIR / 'pendulum'
PENDULUM_PROCESSED_DIR = PENDULUM_DATA_DIR / 'processed'

# 2. Physics Parameters for the Double Pendulum
G = 9.81  # Acceleration due to gravity (m/s^2)
L1 = 1.0  # Length of the first pendulum arm (m)
L2 = 1.0  # Length of the second pendulum arm (m)
M1 = 1.0  # Mass of the first bob (kg)
M2 = 1.0  # Mass of the second bob (kg)

# 3. Simulation & Data Generation Parameters
NUM_TRAJECTORIES = 50      # Generate 50 different simulations
T_MAX = 100.0              # Simulate for 100 seconds
MIN_DT = 0.05              # Minimum time step for irregular sampling
MAX_DT = 0.5               # Maximum time step (10x min), creating significant irregularity

# 4. Machine Learning Parameters
SEQUENCE_LENGTH = 25       # Number of past steps to use for prediction
OUTPUT_DIM = 4             # The state vector: [theta1, omega1, theta2, omega2]

def double_pendulum_ode(t, y):
    """
    Defines the differential equations for the double pendulum.
    State vector y = [theta1, omega1, theta2, omega2]
    """
    theta1, omega1, theta2, omega2 = y
    
    delta = theta2 - theta1
    
    den1 = (M1 + M2) * L1 - M2 * L1 * np.cos(delta) * np.cos(delta)
    num1 = (M2 * L1 * omega1 * omega1 * np.sin(delta) * np.cos(delta) +
            M2 * G * np.sin(theta2) * np.cos(delta) +
            M2 * L2 * omega2 * omega2 * np.sin(delta) -
            (M1 + M2) * G * np.sin(theta1))
    domega1_dt = num1 / (den1 + 1e-9) # Add epsilon for stability

    den2 = (L2 / L1) * den1
    num2 = (-M2 * L2 * omega2 * omega2 * np.sin(delta) * np.cos(delta) +
            (M1 + M2) * G * np.sin(theta1) * np.cos(delta) -
            (M1 + M2) * L1 * omega1 * omega1 * np.sin(delta) -
            (M1 + M2) * G * np.sin(theta2))
    domega2_dt = num2 / (den2 + 1e-9) # Add epsilon for stability
    
    return [omega1, domega1_dt, omega2, domega2_dt]

def generate_irregular_trajectory():
    """
    Generates a single, long, irregularly sampled trajectory.
    """
    y0 = [
        np.pi / 2 + np.random.uniform(-0.1, 0.1),
        0.0,
        np.pi + np.random.uniform(-0.1, 0.1),
        0.0
    ]
    
    # --- SỬA LỖI Ở ĐÂY ---
    # Logic tạo time_points đã được sửa để đảm bảo không vượt quá T_MAX
    time_points = [0.0]
    current_time = 0.0
    while current_time < T_MAX:
        dt = np.random.uniform(MIN_DT, MAX_DT)
        next_time = current_time + dt
        if next_time > T_MAX:
            break  # Thoát vòng lặp nếu điểm tiếp theo vượt quá giới hạn
        time_points.append(next_time)
        current_time = next_time
    
    t_eval = np.array(time_points)
    # ---------------------
    
    sol = solve_ivp(
        double_pendulum_ode,
        [0, T_MAX],
        y0,
        t_eval=t_eval,
        method='RK45'
    )
    
    return sol.y.T

def create_windows(trajectories: list, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Create sequence-to-value windows from a list of trajectories."""
    all_sequences = []
    all_targets = []
    
    print("Creating sliding windows from trajectories...")
    for traj in tqdm(trajectories):
        for i in range(len(traj) - seq_len):
            sequence = traj[i : i + seq_len]
            target = traj[i + seq_len]
            all_sequences.append(sequence)
            all_targets.append(target)
            
    return np.array(all_sequences, dtype=np.float32), np.array(all_targets, dtype=np.float32)

def pendulum_preprocess_pipeline():
    """
    Execute the complete preprocessing pipeline for the double pendulum dataset.
    """
    print("--- Starting Double Pendulum Data Generation and Preprocessing ---")
    os.makedirs(PENDULUM_PROCESSED_DIR, exist_ok=True)

    print(f"Generating {NUM_TRAJECTORIES} long, irregular trajectories...")
    trajectories = [generate_irregular_trajectory() for _ in tqdm(range(NUM_TRAJECTORIES))]

    X, y = create_windows(trajectories, SEQUENCE_LENGTH)
    print(f"Successfully created {len(X)} windows.")
    print(f"Shape of feature matrix X: {X.shape}")
    print(f"Shape of target matrix y: {y.shape}")

    print("Splitting data into training and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False # shuffle=False is often better for time series
    )
    print(f"Training set size: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set size: X={X_test.shape}, y={y_test.shape}")

    print("Standardizing data (fitting scaler on training data only)...")
    X_train_reshaped = X_train.reshape(-1, OUTPUT_DIM)
    
    scaler = StandardScaler()
    scaler.fit(X_train_reshaped)
    
    X_train = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    y_train = scaler.transform(y_train)
    
    X_test_reshaped = X_test.reshape(-1, OUTPUT_DIM)
    X_test = scaler.transform(X_test_reshaped).reshape(X_test.shape)
    y_test = scaler.transform(y_test)
    
    print("Standardization completed for both features (X) and targets (y).")

    print(f"Saving processed data to directory: {PENDULUM_PROCESSED_DIR}")
    
    np.save(PENDULUM_PROCESSED_DIR / 'X_train.npy', X_train)
    np.save(PENDULUM_PROCESSED_DIR / 'y_train.npy', y_train)
    np.save(PENDULUM_PROCESSED_DIR / 'X_test.npy', X_test)
    np.save(PENDULUM_PROCESSED_DIR / 'y_test.npy', y_test)
    
    joblib.dump(scaler, PENDULUM_PROCESSED_DIR / 'scaler.pkl')
    
    print("--- Double Pendulum Data Generation Completed Successfully! ---")

if __name__ == '__main__':
    pendulum_preprocess_pipeline()