import torch

from models.LSTM_CNN import LSTM_CNN
from utils.get_dataset import GetDataset

csv = '../data/Test.csv'
df = GetDataset(csv)
model = LSTM_CNN(modeltype='rnn', input_size=9, lstm_hidden_size=5, lstm_layers=5, lstm_output_size=1, kernel_size=3,
                 padding=1, leaky_relu=0.2)
state_dict = torch.load('../outputs/RNN-CNN.pth')
model.load_state_dict(state_dict)
model.eval()
predictions = model(torch.tensor(df.get_data(isReal=True).drop(columns=['Next_Day_Change']).values)[None, :, :].float())
predictions[predictions > 0] = 1
predictions[predictions <= 0] = 0
predictions = predictions.long()
print(df.df.to_string())
print(predictions)
