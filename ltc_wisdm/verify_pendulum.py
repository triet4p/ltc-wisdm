import numpy as np
from tqdm import tqdm

# Import các hàm và hằng số từ script tiền xử lý của bạn
from .pendulum_preprocess import (
    generate_continuous_solution,
    create_windows_precise,
    PREDICTION_HORIZON,
    SEQUENCE_LENGTH,
    T_MAX, # <-- Import T_MAX
    MIN_DT, # <-- Import MIN_DT
    MAX_DT # <-- Import MAX_DT
)

def verify_data_generation_logic(num_windows_to_check: int = 2):
    print("--- Bắt đầu quy trình xác minh dữ liệu ---")
    
    # Bước 1: Tạo "nguồn chân lý"
    print("\n[1/4] Tạo 1 quỹ đạo với lời giải liên tục...")
    solution_object = generate_continuous_solution()
    print(" -> Hoàn thành.")
    
    # Bước 2: Tạo MỘT LẦN DUY NHẤT các dấu thời gian đầu vào
    print("\n[2/4] Tạo một chuỗi dấu thời gian (timestamps) nhất quán...")
    input_times = []
    current_time = 0.0
    while current_time < T_MAX - PREDICTION_HORIZON:
        dt = np.random.uniform(MIN_DT, MAX_DT)
        if current_time + dt > T_MAX - PREDICTION_HORIZON:
            break
        current_time += dt
        input_times.append(current_time)
    print(f" -> Hoàn thành. Đã tạo {len(input_times)} dấu thời gian.")

    # Bước 3: Tạo các cửa sổ (X, y) bằng cách SỬ DỤNG các dấu thời gian đã tạo
    print(f"\n[3/4] Tạo các cửa sổ (X, y) từ quỹ đạo với horizon = {PREDICTION_HORIZON}s...")
    X_generated, y_generated = create_windows_precise(
        [solution_object], 
        SEQUENCE_LENGTH, 
        PREDICTION_HORIZON,
        given_input_times=input_times # <-- TRUYỀN VÀO ĐÂY
    )
    print(f" -> Hoàn thành. Đã tạo {len(X_generated)} cửa sổ.")

    # Bước 4: Kiểm tra từng cửa sổ
    print("\n[4/4] Bắt đầu kiểm tra từng cửa sổ...")
    if len(X_generated) == 0:
        print("Không có cửa sổ nào được tạo. Bỏ qua kiểm tra.")
        return
        
    for i in range(num_windows_to_check):
        idx_to_check = np.random.randint(0, len(X_generated))
        
        print(f"\n--- KIỂM TRA CỬA SỔ #{idx_to_check} ---")
        
        y_from_dataset = y_generated[idx_to_check]
        
        # Bây giờ `input_times` ở đây và `input_times` được dùng để tạo data là MỘT
        sequence_times = input_times[idx_to_check : idx_to_check + SEQUENCE_LENGTH]
        last_time_in_sequence = sequence_times[-1]
        
        expected_target_time = last_time_in_sequence + PREDICTION_HORIZON
        
        y_ground_truth = solution_object.sol(expected_target_time).T.flatten()
        
        # ... (phần print và so sánh không đổi)
        print(f"  - Thời gian của trạng thái cuối cùng trong X: {last_time_in_sequence:.4f}s")
        print(f"  - Chân trời dự đoán (Horizon):               {PREDICTION_HORIZON:.4f}s")
        print(f"  - Thời gian mục tiêu Y (dự kiến):            {expected_target_time:.4f}s")
        print("\n  - Trạng thái Y từ bộ dữ liệu đã tạo:")
        print(f"    {y_from_dataset}")
        print("\n  - Trạng thái Y được tính lại từ 'nguồn chân lý':")
        print(f"    {y_ground_truth}")
        error = np.linalg.norm(y_from_dataset - y_ground_truth)
        print(f"\n  - Sai số (Euclidean distance) giữa hai vector y: {error:.10f}")
        
        if error < 1e-6:
            print("  -> KẾT QUẢ: CHÍNH XÁC. Dữ liệu được tạo đúng như mong đợi.")
        else:
            print("  -> KẾT QUẢ: CẢNH BÁO. Có sự sai khác lớn, cần kiểm tra lại logic.")

if __name__ == '__main__':
    verify_data_generation_logic(num_windows_to_check=2)