import torch
from torch import nn

# this is the sample model given
class WeatherRNN(nn.Module):
    def __init__(self, n_future_days, input_size=1, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.wind = nn.Linear(hidden_size, n_future_days)
        self.precip = nn.Linear(hidden_size, n_future_days)
        self.temp = nn.Linear(hidden_size, n_future_days)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last = out[:, -1, :]
        result = torch.stack([self.wind(last),
                              self.precip(last),
                              self.temp(last)],axis=-1)
        return result

# this is my model
class WeatherLSTM(nn.Module):
    def __init__(self, n_future_days, input_size=3, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, n_future_days * 3)  # 3 features: temp, wind, precip

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # Take the output of the last time step
        return self.fc(last)