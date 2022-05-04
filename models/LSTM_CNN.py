import torch
from torch.nn import Module, LSTM, Linear, Conv1d, GRU, RNN


class LSTM_CNN(Module):
    def __init__(self, modeltype, input_size, lstm_hidden_size, lstm_layers, lstm_output_size, kernel_size, padding, dropout):
        super(LSTM_CNN, self).__init__()
        if modeltype == 'rnn':
            self.lstm = RNN(input_size=input_size, hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers, batch_first=True, dropout=dropout)
        if modeltype == 'lstm':
            self.lstm = LSTM(input_size=input_size, hidden_size=lstm_hidden_size,
                             num_layers=lstm_layers, batch_first=True, dropout=dropout)
        if modeltype == 'gru':
            self.lstm = GRU(input_size=input_size, hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers, batch_first=True, dropout=dropout)
        self.conv1 = Conv1d(lstm_hidden_size, lstm_hidden_size, kernel_size, padding=padding)
        self.linear = Linear(in_features=lstm_hidden_size, out_features=lstm_output_size)

    def forward(self, out):
        out, _ = self.lstm(out)
        out = out[:, -1, :].permute(1, 0)
        out = torch.unsqueeze(out, dim=0)
        out = self.conv1(out)
        out = torch.squeeze(out, dim=0)
        out = out.permute(1, 0)
        linear_out = self.linear(out)
        out = linear_out
        return out
