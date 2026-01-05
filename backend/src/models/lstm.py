import torch
import torch.nn as nn


class M5LSTM(nn.Module):
    def __init__(self, num_features, hidden_dim=64, num_layers=2, dropout=0.2):
        """
        Standard LSTM for Time Series Forecasting.
        """
        super(M5LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM Layer
        # input_size = number of features per time step
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Fully Connected Output Layer
        # Maps the final hidden state to a single sales prediction
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)

        # Initialize hidden and cell states (optional, PyTorch does this by default)
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Forward propagate LSTM
        # out shape: (batch_size, seq_len, hidden_dim)
        out, _ = self.lstm(x)

        # We only care about the output of the LAST time step
        last_time_step = out[:, -1, :]  # Shape: (batch_size, hidden_dim)

        # Decode the hidden state of the last time step
        prediction = self.fc(last_time_step)  # Shape: (batch_size, 1)

        return prediction.squeeze()  # Return flat vector
