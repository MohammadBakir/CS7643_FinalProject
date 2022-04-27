from torch import nn
from torch.nn import Module, Linear, LeakyReLU, GRU, RNN


class LSTM(Module):
    def __init__(self, modeltype, input_size, lstm_hidden_size, lstm_layers, lstm_output_size, leaky_relu):
        super(LSTM, self).__init__()
        if modeltype == 'rnn':
            self.lstm = RNN(input_size=input_size, hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers, batch_first=True)
        if modeltype == 'lstm':
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_hidden_size,
                                num_layers=lstm_layers, batch_first=True)
        if modeltype == 'gru':
            self.lstm = GRU(input_size=input_size, hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers, batch_first=True)
        self.linear = Linear(in_features=lstm_hidden_size, out_features=lstm_output_size)
        self.leaky_relu = LeakyReLU(leaky_relu)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        out = self.leaky_relu(out)
        return out
