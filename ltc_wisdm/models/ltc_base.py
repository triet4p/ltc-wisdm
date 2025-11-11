# src/models/ltc_base.py

import random
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
    def __init__(self, input_dim: int, hidden_dim: int):
        super(LTCCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Core layers for dynamics
        self.gates_and_dynamics = nn.Linear(input_dim + hidden_dim, 2 * hidden_dim)
        self.tau_linear = nn.Linear(hidden_dim, hidden_dim)

        # ==========================================================
        # <<< STABILITY MECHANISM 1: Stabilizing Parameters >>>
        # Introduce gleak and cm as learnable parameters, inspired by ncps.
        # These will stabilize the denominator of the ODE.
        # ==========================================================
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

    def forward(self, t: float, h: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """
        Defines the stabilized differential equation dh/dt.
        """
        combined = torch.cat([x_t, h], dim=1)
        g_d_combined = self.gates_and_dynamics(combined)
        gate, dynamics = torch.chunk(g_d_combined, 2, dim=1)
        sigmoid_gate = torch.sigmoid(gate)
        tanh_dynamics = torch.tanh(dynamics)
        
        # Original tau calculation (dependent on h)
        tau = F.softplus(self.tau_linear(h))
        
        # ==========================================================
        # <<< STABILITY MECHANISM: Stabilized ODE >>>
        # The numerator remains the same, representing the driving force.
        # The denominator is now stabilized by `cm` and `gleak`.
        # We use softplus to ensure they are always positive at runtime.
        # ==========================================================
        numerator = sigmoid_gate * tanh_dynamics - h
        denominator = tau + F.softplus(self.cm) + F.softplus(self.gleak)
        
        dhdt = numerator / (denominator + 1e-6) # Add epsilon for safety
        
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
        
        self.cell = LTCCell(input_dim, hidden_dim, tc)
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