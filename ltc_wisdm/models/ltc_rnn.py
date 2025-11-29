from typing import Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

class LTCRNNCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, rnn_hidden_dim: int):
        super(LTCRNNCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        # Lớp Linear bây giờ nhận 3 đầu vào: input, h_ltc, và h_rnn
        total_input_dim = input_dim + hidden_dim + rnn_hidden_dim
        self.gates_and_dynamics = nn.Linear(total_input_dim, 2 * hidden_dim)
        self.tau_linear = nn.Linear(hidden_dim, hidden_dim)

        self.gleak = nn.Parameter(torch.Tensor(hidden_dim))
        self.cm = nn.Parameter(torch.Tensor(hidden_dim))
        self.init_parameters()
        
    def init_parameters(self):
        """
        Apply constrained initialization to all parameters.
        """
        # ==========================================================
        # <<< STABILITY MECHANISM 2: Constrained Initialization >>>
        # Initialize weights and biases in small, controlled ranges.
        # ==========================================================
        with torch.no_grad():
            # For main dynamics layer
            torch.nn.init.uniform_(self.gates_and_dynamics.weight, -0.1, 0.1)
            torch.nn.init.uniform_(self.gates_and_dynamics.bias, -0.1, 0.1)
            # For tau layer
            torch.nn.init.uniform_(self.tau_linear.weight, -0.1, 0.1)
            torch.nn.init.uniform_(self.tau_linear.bias, -0.1, 0.1)
            # For stabilizing parameters (inspired by ncps init_ranges)
            torch.nn.init.uniform_(self.gleak, 0.001, 1.0)
            torch.nn.init.uniform_(self.cm, 0.4, 0.6)
            
    def forward(self, t: float, h_ltc: torch.Tensor, x_t: torch.Tensor, h_rnn: torch.Tensor) -> torch.Tensor:
        # Nối cả 3 tensor lại với nhau
        combined = torch.cat([x_t, h_ltc, h_rnn], dim=1)
        
        g_d_combined = self.gates_and_dynamics(combined)
        gate, dynamics = torch.chunk(g_d_combined, 2, dim=1)
        # ... (phần còn lại của hàm forward giống hệt LTCCell)
        sigmoid_gate = torch.sigmoid(gate)
        tanh_dynamics = torch.tanh(dynamics)
        tau = F.softplus(self.tau_linear(h_ltc))
        numerator = sigmoid_gate * tanh_dynamics - h_ltc
        denominator = tau + F.softplus(self.cm) + F.softplus(self.gleak)
        dhdt = numerator / (denominator + 1e-6)
        return dhdt
    
class RNNODEFunc(nn.Module):
    def __init__(self, cell: LTCRNNCell, x_t: torch.Tensor, h_rnn: torch.Tensor):
        super(RNNODEFunc, self).__init__()
        self.cell = cell
        self.x_t = x_t
        self.h_rnn = h_rnn

    def forward(self, t: float, h_ltc: torch.Tensor) -> torch.Tensor:
        return self.cell(t, h_ltc, self.x_t, self.h_rnn)

class LTCRNNModel(nn.Module):
    def __init__(self, input_dim: int, ltc_hidden_dim: int, rnn_hidden_dim: int, output_dim: int, solver: str = 'rk4',
                 rnn_cell_cls: Type[nn.modules.rnn.RNNCellBase] = nn.LSTMCell):
        super(LTCRNNModel, self).__init__()
        self.ltc_hidden_dim = ltc_hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        
        # 1. Thêm một RNN (ví dụ GRU)
        self.rnn = rnn_cell_cls(input_size=input_dim, hidden_size=rnn_hidden_dim)
        
        # 2. Sử dụng LTCMixedCell mới
        self.cell = LTCRNNCell(input_dim, ltc_hidden_dim, rnn_hidden_dim)
        self.fc = nn.Linear(ltc_hidden_dim, output_dim)
        
        self.solver = solver
        self.odeint_fn = odeint
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        # Khởi tạo cả hai trạng thái ẩn
        h_ltc = torch.zeros(batch_size, self.ltc_hidden_dim, device=x.device)
        h_rnn = torch.zeros(batch_size, self.rnn_hidden_dim, device=x.device)
        integration_time = self.integration_time.to(x.device)

        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Chạy RNN trước để có "tín hiệu điều khiển"
            h_rnn = self.rnn(x_t, h_rnn)
            
            # Tạo ODE function với cả x_t và h_rnn
            ode_func = RNNODEFunc(self.cell, x_t, h_rnn)
            
            h_traj = self.odeint_fn(ode_func, h_ltc, integration_time, method=self.solver)
            h_ltc = h_traj[-1]
        
        out = self.fc(h_ltc)
        return out