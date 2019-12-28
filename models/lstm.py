"""
This module defines a generic LSTM model.
"""

# Externals
import torch
import torch.nn as nn

class LSTM(nn.Module):
    """
    Generic RNN classifier with LSTM cell and dense linear output.
    """
    def __init__(self, input_size, hidden_size, output_size=1, n_lstm_layers=1):
        """Model constructor"""
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=n_lstm_layers,
                           batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        state_shape = (self.rnn.num_layers, x.size(0), self.rnn.hidden_size)
        h0 = torch.zeros(*state_shape, dtype=torch.float, device=x.device)
        c0 = torch.zeros(*state_shape, dtype=torch.float, device=x.device)
        o, (h, c) = self.rnn(x, (h0, c0))
        return self.linear(h[-1])
