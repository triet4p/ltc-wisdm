import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

class LTCAttentionCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, attention_dim: int):
        super(LTCAttentionCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim # Kích thước của context vector

        # Linear layer nhận: input + hidden_state + context_vector
        total_input_dim = input_dim + hidden_dim + attention_dim
        
        self.gates_and_dynamics = nn.Linear(total_input_dim, 2 * hidden_dim)
        self.tau_linear = nn.Linear(hidden_dim, hidden_dim)

        self.gleak = nn.Parameter(torch.Tensor(hidden_dim))
        self.cm = nn.Parameter(torch.Tensor(hidden_dim))
        
        self.init_parameters()

    def init_parameters(self):
        # Khởi tạo giống hệt các phiên bản trước để đảm bảo công bằng
        with torch.no_grad():
            torch.nn.init.uniform_(self.gates_and_dynamics.weight, -0.1, 0.1)
            torch.nn.init.uniform_(self.gates_and_dynamics.bias, -0.1, 0.1)
            torch.nn.init.uniform_(self.tau_linear.weight, -0.1, 0.1)
            torch.nn.init.uniform_(self.tau_linear.bias, -0.1, 0.1)
            torch.nn.init.uniform_(self.gleak, 0.001, 1.0)
            torch.nn.init.uniform_(self.cm, 0.4, 0.6)

    def forward(self, t: float, h_ltc: torch.Tensor, x_t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Nối: [Input hiện tại, Trạng thái LTC hiện tại, Context từ quá khứ]
        combined = torch.cat([x_t, h_ltc, context], dim=1)
        
        g_d_combined = self.gates_and_dynamics(combined)
        gate, dynamics = torch.chunk(g_d_combined, 2, dim=1)
        
        sigmoid_gate = torch.sigmoid(gate)
        tanh_dynamics = torch.tanh(dynamics)
        tau = F.softplus(self.tau_linear(h_ltc))
        
        numerator = sigmoid_gate * tanh_dynamics - h_ltc
        denominator = tau + F.softplus(self.cm) + F.softplus(self.gleak)
        
        dhdt = numerator / (denominator + 1e-6)
        return dhdt

class AttentionODEFunc(nn.Module):
    def __init__(self, cell: LTCAttentionCell, x_t: torch.Tensor, context: torch.Tensor):
        super(AttentionODEFunc, self).__init__()
        self.cell = cell
        self.x_t = x_t
        self.context = context

    def forward(self, t: float, h_ltc: torch.Tensor) -> torch.Tensor:
        return self.cell(t, h_ltc, self.x_t, self.context)
    
class StepwiseAttention(nn.Module):
    """
    Cơ chế Attention nhẹ, tính toán context vector dựa trên lịch sử hidden states.
    """
    def __init__(self, input_dim: int, hidden_dim: int, attention_dim: int):
        super(StepwiseAttention, self).__init__()
        # Query được tạo từ Input hiện tại
        self.W_query = nn.Linear(input_dim, attention_dim)
        # Key và Value được tạo từ Hidden State quá khứ
        self.W_key = nn.Linear(hidden_dim, attention_dim)
        self.W_value = nn.Linear(hidden_dim, attention_dim)
        
        self.scale = torch.sqrt(torch.FloatTensor([attention_dim]))

    def forward(self, x_t: torch.Tensor, history_h: torch.Tensor):
        """
        Args:
            x_t: (batch, input_dim) - Input hiện tại
            history_h: (batch, seq_len_so_far, hidden_dim) - Lịch sử các trạng thái
        """
        if history_h is None or history_h.size(1) == 0:
            # Nếu chưa có lịch sử, trả về vector 0
            return torch.zeros(x_t.size(0), self.W_value.out_features, device=x_t.device)

        # 1. Tính Query từ input hiện tại
        # Q shape: (batch, 1, att_dim)
        Q = self.W_query(x_t).unsqueeze(1)
        
        # 2. Tính Key và Value từ lịch sử
        # K, V shape: (batch, seq_len, att_dim)
        K = self.W_key(history_h)
        V = self.W_value(history_h)
        
        # 3. Dot Product Attention: score = Q * K.T
        # scores shape: (batch, 1, seq_len)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale.to(x_t.device)
        
        # 4. Softmax để lấy trọng số
        attn_weights = F.softmax(scores, dim=-1)
        
        # 5. Context = weights * V
        # context shape: (batch, 1, att_dim) -> squeeze -> (batch, att_dim)
        context = torch.bmm(attn_weights, V).squeeze(1)
        
        return context
    
class LTCAttentionModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 attention_dim: int = 32, solver: str = 'rk4'):
        super(LTCAttentionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        
        # 1. Attention Mechanism
        self.attention = StepwiseAttention(input_dim, hidden_dim, attention_dim)
        
        # 2. LTC Cell với Attention input
        self.cell = LTCAttentionCell(input_dim, hidden_dim, attention_dim)
        
        # 3. Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        self.solver = solver
        self.odeint_fn = odeint

    def forward(self, x_tuple: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Hỗ trợ input dạng tuple (states, dts) cho bài toán con lắc.
        Nếu input chỉ là tensor (bài toán WISDM), nó tự xử lý.
        """
        # Xử lý linh hoạt đầu vào (cho cả WISDM và Pendulum)
        if isinstance(x_tuple, tuple):
            x_states, x_dts = x_tuple
        else:
            x_states = x_tuple
            # Nếu không có dt, giả định dt=1 (hoặc một giá trị cố định nhỏ)
            x_dts = torch.ones(x_states.shape[0], x_states.shape[1], device=x_states.device) * 0.05

        batch_size, seq_len, _ = x_states.shape
        
        # Khởi tạo hidden state
        h = torch.zeros(batch_size, self.hidden_dim, device=x_states.device)
        
        # Bộ nhớ lịch sử cho Attention: Danh sách các tensor (batch, hidden_dim)
        history_list = [] 

        for t in range(seq_len):
            x_t = x_states[:, t, :]
            dt = x_dts[:, t]
            
            # --- BƯỚC 1: Tính Context từ Attention ---
            # Chuyển list lịch sử thành tensor (batch, t, hidden_dim)
            if len(history_list) > 0:
                history_tensor = torch.stack(history_list, dim=1)
            else:
                history_tensor = None
                
            context = self.attention(x_t, history_tensor)
            
            # --- BƯỚC 2: Tích phân ODE ---
            # (Logic xử lý dt biến thiên trong batch)
            # Tạo integration time chuẩn hóa
            integration_time = torch.tensor([0, 1]).float().to(x_states.device)
            
            # Lưu ý: Để chính xác tuyệt đối với dt biến thiên trong batch, ta nên loop qua batch
            # hoặc scale gradient. Ở đây ta dùng cách scale đơn giản để tối ưu tốc độ:
            # dh/dt = f(...) * dt_scale.
            # Tuy nhiên, để nhất quán với code cũ, ta sẽ dùng cách loop đơn giản nếu cần
            # hoặc truyền dt vào ODE func.
            
            # Cách đơn giản và hiệu quả: Scale vector thời gian
            # Chúng ta tích phân từ 0 đến 1, nhưng nhân đạo hàm với dt thực tế.
            # Hoặc đơn giản hơn: Tích phân từ 0 đến dt_trung_bình (như code cũ)
            # Ở đây tôi dùng cách tiếp cận từng mẫu trong batch để chính xác nhất cho Attention
            
            ode_func = AttentionODEFunc(self.cell, x_t, context)
            
            next_h_list = []
            for i in range(batch_size):
                t_span = torch.tensor([0, dt[i]]).float().to(x_states.device)
                h_i = h[i].unsqueeze(0)
                # Cần func riêng cho từng mẫu vì x_t[i] và context[i] khác nhau
                # (Lưu ý: Để tối ưu tốc độ thực sự, cần viết lại ODEFunc hỗ trợ batch dt, 
                # nhưng code này ưu tiên tính chính xác logic)
                
                # Để tránh tạo quá nhiều object, ta dùng 1 trick:
                # Tích phân từ 0 -> 1, và trong cell ta nhân dhdt với dt[i]
                # Nhưng để an toàn với solver RK4, ta dùng vòng lặp tường minh này:
                
                # Tạo wrapper nhỏ để giữ x_t[i] và context[i]
                class SingleSampleODE(nn.Module):
                    def __init__(self, parent_cell, xi, ci):
                        super().__init__()
                        self.cell = parent_cell
                        self.xi, self.ci = xi, ci
                    def forward(self, t, hi):
                        return self.cell(t, hi, self.xi, self.ci)
                
                sample_func = SingleSampleODE(self.cell, x_t[i].unsqueeze(0), context[i].unsqueeze(0))
                h_traj_i = self.odeint_fn(sample_func, h_i, t_span, method=self.solver)
                next_h_list.append(h_traj_i[-1])
            
            h = torch.cat(next_h_list, dim=0)
            
            # Lưu trạng thái vào lịch sử
            history_list.append(h)
        
        out = self.fc(h)
        return out