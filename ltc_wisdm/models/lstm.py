# src/models/lstm.py

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    LSTM model for sequence classification.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 1, dropout: float = 0.2):
        """
        Initialize LSTM architecture.

        Args:
            input_dim (int): Number of input features at each time step (number of sensor axes).
            hidden_dim (int): Size of hidden state.
            num_classes (int): Number of output classes (number of activities).
            num_layers (int, optional): Number of stacked LSTM layers. Defaults to 1.
            dropout (float, optional): Dropout rate between LSTM layers (if num_layers > 1). Defaults to 0.2.
        """
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer:
        # - batch_first=True: Requires input with shape (batch, seq_len, features)
        # - dropout: Applies dropout between LSTM layers if more than 1 layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Final fully connected layer for classification:
        # Takes the final hidden state of the LSTM as input and outputs logits for classes.
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: Output logits with shape (batch_size, num_classes).
        """
        # Initialize hidden state and cell state with zeros
        # h0 shape: (num_layers, batch_size, hidden_dim)
        # c0 shape: (num_layers, batch_size, hidden_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Pass data through LSTM
        # lstm_out shape: (batch_size, sequence_length, hidden_dim)
        # h_n shape: (num_layers, batch_size, hidden_dim)
        # c_n shape: (num_layers, batch_size, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))
        
        # We only need the final hidden state of the top LSTM layer for classification.
        # h_n[-1] will extract the hidden state of the last layer, shape: (batch_size, hidden_dim)
        last_hidden_state = h_n[-1]
        
        # Pass the final hidden state through the fully connected layer
        out = self.fc(last_hidden_state)
        
        return out