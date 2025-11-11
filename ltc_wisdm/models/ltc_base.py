# src/models/ltc_base.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

class LTCCell(nn.Module):
    """
    An upgraded Liquid Time-Constant (LTC) cell.
    
    This implementation is inspired by the official `ncps` library, featuring:
    - A more complex gating mechanism by mixing inputs and hidden states.
    - A dynamic time-constant `tau` for adaptive temporal processing.
    
    It is designed to be used with an ODE solver like torchdiffeq.
    """
    def __init__(self, input_dim: int, hidden_dim: int, tc: float, debug: bool = True):
        super(LTCCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # A single, larger Linear layer to compute gates and dynamics together.
        # Input is a concatenation of [x_t, h_t], so size = (input_dim + hidden_dim)
        # Output provides for both the gate and the dynamics, so size = (2 * hidden_dim)
        self.gates_and_dynamics = nn.Linear(input_dim + hidden_dim, 2 * hidden_dim)
        
        # A separate Linear layer to compute the adaptive time-constant `tau`.
        self.tau_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.tc = tc
        self.debug = debug
        
    def _print_debug(self, x):
        if self.debug:
            print(x)

    def forward(self, t: float, h: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """
        Defines the differential equation dh/dt = f(h, x, t).

        Args:
            t (float): Current time (required by odeint, but not used in this simple case).
            h (torch.Tensor): Current hidden state of shape (batch_size, hidden_dim).
            x_t (torch.Tensor): Input at the current time step of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: The derivative of the hidden state (dh/dt) of shape (batch_size, hidden_dim).
        """
        if not torch.isfinite(h).all():
            self._print_debug(f"!!! DETECTED NaN/Inf IN INPUT 'h' TO THE CELL !!!")
            # Trả về zero grad để dừng vòng lặp một cách an toàn
            return torch.zeros_like(h)
        
        # 1. Concatenate input and hidden state
        combined = torch.cat([x_t, h], dim=1)
        
        # 2. Compute gates and dynamics simultaneously
        g_d_combined = self.gates_and_dynamics(combined)
        
        # 3. Split the result into two parts: gate and dynamics
        gate, dynamics = torch.chunk(g_d_combined, 2, dim=1)
        
        # 4. Apply activation functions
        sigmoid_gate = torch.sigmoid(gate)
        tanh_dynamics = torch.tanh(dynamics)
        
        # 5. Compute the adaptive time-constant `tau`
        # Using softplus to ensure tau is always positive
        tau = F.softplus(self.tau_linear(h)) + self.tc
        
        if torch.randint(0, 100, (1,)).item() == 0:
            self._print_debug(f"h norm: {h.norm().item():.2f} | "
                   f"tau_min: {tau.min().item():.4f} | "
                   f"tau_mean: {tau.mean().item():.4f}")
        
        # 6. Apply the gated differential equation
        # dh/dt = (gate * dynamics - h) / tau
        dhdt = (sigmoid_gate * tanh_dynamics - h) / tau
        if not torch.isfinite(dhdt).all():
            self._print_debug(f"!!! NaN/Inf DETECTED IN 'dhdt' OUTPUT !!!")
            self._print_debug(f"  - sigmoid_gate norm: {sigmoid_gate.norm().item():.2f}")
            self._print_debug(f"  - tanh_dynamics norm: {tanh_dynamics.norm().item():.2f}")
            self._print_debug(f"  - h norm: {h.norm().item():.2f}")
            self._print_debug(f"  - tau min: {tau.min().item():.4f}, tau mean: {tau.mean().item():.4f}")
            # Trả về zero grad để dừng lại
            return torch.zeros_like(h)
        
        return dhdt

class ODEFunc(nn.Module):
    """A helper class to wrap the LTCCell for the ODE solver."""
    def __init__(self, cell: LTCCell, x_t: torch.Tensor):
        super(ODEFunc, self).__init__()
        self.cell = cell
        self.x_t = x_t

    def forward(self, t: float, h: torch.Tensor) -> torch.Tensor:
        return self.cell(t, h, self.x_t)

class LTCModel(nn.Module):
    """
    A full Liquid Time-Constant (LTC) model for sequence classification.
    
    This model uses an ODE solver (`torchdiffeq`) to process sequences step-by-step.
    """
    def __init__(self, input_dim: int, hidden_dim: int, 
                 num_classes: int, 
                 tc: float,
                 solver: str = 'rk4', 
                 use_adjoint: bool = False,
                 debug: bool = True):
        """
        Initializes the LTC model architecture.

        Args:
            input_dim (int): Number of input features at each time step.
            hidden_dim (int): The size of the hidden state.
            num_classes (int): Number of output classes.
            solver (str, optional): The ODE solver to use (e.g., 'euler', 'rk4'). Defaults to 'rk4'.
            use_adjoint (bool, optional): Whether to use the memory-efficient adjoint method for backpropagation. Defaults to False.
        """
        super(LTCModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        self.cell = LTCCell(input_dim, hidden_dim, tc, debug)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
        self.solver = solver
        self.odeint_fn = odeint
        # Note: Adjoint method can be slower but saves memory. For this project, standard odeint is fine.
        # self.odeint_fn = odeint_adjoint if use_adjoint else odeint
        
        # Time interval for the ODE solver to integrate over for each step.
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        integration_time = self.integration_time.to(x.device)

        # We need to loop through the sequence manually because the ODE solver
        # processes one step at a time, with the input `x_t` as a parameter of the ODE function.
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Create a new ODE function for each time step, "freezing" the current input x_t
            ode_func = ODEFunc(self.cell, x_t)
            
            # Solve the ODE for the current time step
            h_traj = self.odeint_fn(ode_func, h, integration_time, method=self.solver)
            
            # The new hidden state is the last point of the trajectory
            h = h_traj[-1]
        
        # Use the final hidden state for classification
        out = self.fc(h)
        return out