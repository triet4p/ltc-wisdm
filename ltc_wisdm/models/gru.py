# src/models/gru.py

import torch
import torch.nn as nn

class GRUModel(nn.Module):
    """
    GRU model for sequence classification.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 1, dropout: float = 0.2):
        """
        Initialize GRU architecture.

        Args:
            input_dim (int): Number of input features at each time step (number of sensor axes).
            hidden_dim (int): Size of hidden state.
            output_dim (int): Number of output classes (number of activities).
            num_layers (int, optional): Number of stacked GRU layers. Defaults to 1.
            dropout (float, optional): Dropout rate between GRU layers (if num_layers > 1). Defaults to 0.2.
        """
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU layer:
        # Similar to LSTM but without cell state.
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Final fully connected layer for classification.
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: Output logits with shape (batch_size, output_dim).
        """
        # Initialize hidden state with zeros
        # h0 shape: (num_layers, batch_size, hidden_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Pass data through GRU
        # gru_out shape: (batch_size, sequence_length, hidden_dim)
        # h_n shape: (num_layers, batch_size, hidden_dim)
        gru_out, h_n = self.gru(x, h0)

        # Get the last hidden state of the top layer for classification
        # h_n[-1] shape: (batch_size, hidden_dim)
        last_hidden_state = h_n[-1]

        # Pass through fully connected layer
        out = self.fc(last_hidden_state)
        
        return out