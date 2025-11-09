# src/models/lstm.py

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    Mô hình LSTM cho bài toán phân loại chuỗi (Sequence Classification).
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 1, dropout: float = 0.2):
        """
        Khởi tạo kiến trúc LSTM.

        Args:
            input_dim (int): Số lượng features đầu vào tại mỗi bước thời gian (số trục cảm biến).
            hidden_dim (int): Kích thước của trạng thái ẩn.
            num_classes (int): Số lượng lớp đầu ra (số loại hoạt động).
            num_layers (int, optional): Số lượng lớp LSTM xếp chồng. Defaults to 1.
            dropout (float, optional): Tỷ lệ dropout giữa các lớp LSTM (nếu num_layers > 1). Defaults to 0.2.
        """
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Lớp LSTM:
        # - batch_first=True: Yêu cầu input có shape (batch, seq_len, features)
        # - dropout: Áp dụng dropout giữa các lớp LSTM nếu có nhiều hơn 1 lớp
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Lớp Fully Connected cuối cùng để phân loại:
        # Nhận đầu vào là trạng thái ẩn cuối cùng của LSTM và đưa ra logits cho các lớp.
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass của mô hình.

        Args:
            x (torch.Tensor): Tensor đầu vào có shape (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: Logits đầu ra có shape (batch_size, num_classes).
        """
        # Khởi tạo trạng thái ẩn và cell state ban đầu bằng zero
        # h0 shape: (num_layers, batch_size, hidden_dim)
        # c0 shape: (num_layers, batch_size, hidden_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Cho dữ liệu đi qua LSTM
        # lstm_out shape: (batch_size, sequence_length, hidden_dim)
        # h_n shape: (num_layers, batch_size, hidden_dim)
        # c_n shape: (num_layers, batch_size, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))
        
        # Chúng ta chỉ cần trạng thái ẩn cuối cùng của lớp LSTM trên cùng để phân loại.
        # h_n[-1] sẽ lấy ra trạng thái ẩn của lớp cuối cùng, shape: (batch_size, hidden_dim)
        last_hidden_state = h_n[-1]
        
        # Đưa trạng thái ẩn cuối cùng qua lớp fully connected
        out = self.fc(last_hidden_state)
        
        return out