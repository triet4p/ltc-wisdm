import os
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

# --- Constants Configuration ---
# (Giữ nguyên các hằng số bất đối xứng và khó)
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / 'data'
PENDULUM_DATA_DIR = DATA_DIR / 'pendulum'
PENDULUM_PROCESSED_DIR = PENDULUM_DATA_DIR / 'processed'

G = 9.81
L1 = 1.0
L2 = 1.8
M1 = 2.0
M2 = 0.5

NUM_TRAJECTORIES = 150
T_MAX = 100.0
MIN_DT = 0.05
MAX_DT = 1.5

SEQUENCE_LENGTH = 15
OUTPUT_DIM = 4
PREDICTION_HORIZON = 1.5

def double_pendulum_ode(t, y):
    # ... (hàm này không đổi)
    theta1, omega1, theta2, omega2 = y
    delta = theta2 - theta1
    den1 = (M1 + M2) * L1 - M2 * L1 * np.cos(delta) * np.cos(delta)
    num1 = (M2 * L1 * omega1 * omega1 * np.sin(delta) * np.cos(delta) +
            M2 * G * np.sin(theta2) * np.cos(delta) +
            M2 * L2 * omega2 * omega2 * np.sin(delta) -
            (M1 + M2) * G * np.sin(theta1))
    domega1_dt = num1 / (den1 + 1e-9)
    den2 = (L2 / L1) * den1
    num2 = (-M2 * L2 * omega2 * omega2 * np.sin(delta) * np.cos(delta) +
            (M1 + M2) * G * np.sin(theta1) * np.cos(delta) -
            (M1 + M2) * L1 * omega1 * omega1 * np.sin(delta) -
            (M1 + M2) * G * np.sin(theta2))
    domega2_dt = num2 / (den2 + 1e-9)
    return [omega1, domega1_dt, omega2, domega2_dt]

def generate_continuous_solution():
    """
    Tạo ra một đối tượng lời giải liên tục cho một quỹ đạo.
    """
    y0 = [np.pi / 2 + np.random.uniform(-0.1, 0.1), 0.0, np.pi + np.random.uniform(-0.1, 0.1), 0.0]
    
    # Chạy solver mà không có t_eval để có được hàm nội suy
    sol = solve_ivp(
        double_pendulum_ode,
        [0, T_MAX],
        y0,
        dense_output=True, # Yêu cầu solver trả về hàm nội suy
        method='RK45'
    )
    return sol

def create_windows_precise(solutions: list, seq_len: int, horizon: float, given_input_times: list = None):
    """
    Tạo cửa sổ (X, y) với y là trạng thái tại một thời điểm tương lai CHÍNH XÁC.
    Có thể nhận một danh sách các input_times đã được tạo trước để đảm bảo tính nhất quán.
    """
    all_sequences = []
    all_targets = []
    
    print(f"Creating windows with a PRECISE prediction horizon of {horizon}s...")
    for sol in tqdm(solutions):
        # Nếu không có input_times nào được cung cấp, hãy tạo chúng ngẫu nhiên
        if given_input_times is None:
            input_times = []
            current_time = 0.0
            while current_time < sol.sol.t[-1] - horizon: # Sử dụng sol.sol.t[-1] để lấy T_MAX thực tế
                dt = np.random.uniform(MIN_DT, MAX_DT)
                if current_time + dt > sol.sol.t[-1] - horizon:
                    break
                current_time += dt
                input_times.append(current_time)
        # Ngược lại, sử dụng danh sách đã được cung cấp
        else:
            input_times = given_input_times

        # Phần còn lại của hàm không thay đổi
        for i in range(len(input_times) - seq_len):
            sequence_times = input_times[i : i + seq_len]
            last_time_in_sequence = sequence_times[-1]
            target_time = last_time_in_sequence + horizon
            
            sequence_states = sol.sol(sequence_times).T
            target_state = sol.sol(target_time).T.flatten()
            
            all_sequences.append(sequence_states)
            all_targets.append(target_state)
            
    return np.array(all_sequences, dtype=np.float32), np.array(all_targets, dtype=np.float32)

def pendulum_preprocess_pipeline():
    print("--- Starting PRECISE Double Pendulum Data Generation ---")
    os.makedirs(PENDULUM_PROCESSED_DIR, exist_ok=True)

    print(f"Generating {NUM_TRAJECTORIES} continuous solution objects (this may take a while)...")
    solutions = [generate_continuous_solution() for _ in tqdm(range(NUM_TRAJECTORIES))]

    X, y = create_windows_precise(solutions, SEQUENCE_LENGTH, PREDICTION_HORIZON)
    
    print(f"Successfully created {len(X)} precise windows.")
    print(f"Shape of feature matrix X: {X.shape}")
    print(f"Shape of target matrix y: {y.shape}")

    # Phần còn lại của pipeline không thay đổi
    print("Splitting data into training and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"Training set size: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set size: X={X_test.shape}, y={y_test.shape}")
    
    print("Standardizing data...")
    X_train_reshaped = X_train.reshape(-1, OUTPUT_DIM)
    scaler = StandardScaler()
    scaler.fit(X_train_reshaped)
    X_train = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    y_train = scaler.transform(y_train)
    X_test_reshaped = X_test.reshape(-1, OUTPUT_DIM)
    X_test = scaler.transform(X_test_reshaped).reshape(X_test.shape)
    y_test = scaler.transform(y_test)
    
    print("Standardization completed.")
    print(f"Saving processed data to directory: {PENDULUM_PROCESSED_DIR}")
    np.save(PENDULUM_PROCESSED_DIR / 'X_train.npy', X_train)
    np.save(PENDULUM_PROCESSED_DIR / 'y_train.npy', y_train)
    np.save(PENDULUM_PROCESSED_DIR / 'X_test.npy', X_test)
    np.save(PENDULUM_PROCESSED_DIR / 'y_test.npy', y_test)
    joblib.dump(scaler, PENDULUM_PROCESSED_DIR / 'scaler.pkl')
    
    print("--- PRECISE Double Pendulum Data Generation Completed Successfully! ---")

if __name__ == '__main__':
    pendulum_preprocess_pipeline()