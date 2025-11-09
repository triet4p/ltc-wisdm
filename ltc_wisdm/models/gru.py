# src/models/gru.py

import torch
import torch.nn as nn

class GRUModel(nn.Module):
    """
    Mô hình GRU cho bài toán phân loại chuỗi (Sequence Classification).
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 1, dropout: float = 0.2):
        """
        Khởi tạo kiến trúc GRU.

        Args:
            input_dim (int): Số lượng features đầu vào tại mỗi bước thời gian (số trục cảm biến).
            hidden_dim (int): Kích thước của trạng thái ẩn.
            num_classes (int): Số lượng lớp đầu ra (số loại hoạt động).
            num_layers (int, optional): Số lượng lớp GRU xếp chồng. Defaults to 1.
            dropout (float, optional): Tỷ lệ dropout giữa các lớp GRU (nếu num_layers > 1). Defaults to 0.2.
        """
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Lớp GRU:
        # Tương tự như LSTM nhưng không có cell state.
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Lớp Fully Connected cuối cùng để phân loại.
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass của mô hình.

        Args:
            x (torch.Tensor): Tensor đầu vào có shape (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: Logits đầu ra có shape (batch_size, num_classes).
        """
        # Khởi tạo trạng thái ẩn ban đầu bằng zero
        # h0 shape: (num_layers, batch_size, hidden_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Cho dữ liệu đi qua GRU
        # gru_out shape: (batch_size, sequence_length, hidden_dim)
        # h_n shape: (num_layers, batch_size, hidden_dim)
        gru_out, h_n = self.gru(x, h0)

        # Lấy trạng thái ẩn cuối cùng của lớp trên cùng để phân loại
        # h_n[-1] shape: (batch_size, hidden_dim)
        last_hidden_state = h_n[-1]

        # Đưa qua lớp fully connected
        out = self.fc(last_hidden_state)
        
        return out