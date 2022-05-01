import torch
from torch.utils.data import DataLoader
from pandas import Series

from models.LSTM_CNN import LSTM_CNN
from utils.get_dataset import GetDataset, StockData

csv = '../data/Test.csv'
device = 'cpu'
num_days = 3
df = GetDataset(csv)
model = LSTM_CNN(modeltype='rnn', input_size=6, lstm_hidden_size=5, lstm_layers=5, lstm_output_size=1, kernel_size=3,
                 padding=1)
state_dict = torch.load('../outputs/LSTM-CNN_SPX.pth')
model.load_state_dict(state_dict)
model.eval()
data = df.get_data()
dataset = StockData(data.to_numpy(), num_days=num_days)
test_dataloader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False)

predictions = []
with torch.no_grad():
    for batch_idx, (data_to_model, target) in enumerate(test_dataloader):
        data_to_model, target = data_to_model.to(device), target.to(device)
        out = model(data_to_model.float())
        predictions.append(torch.round(torch.sigmoid(out)).item())

print(df.df.to_string())
df.df = df.df[num_days - 1:]
df.df['TrueLabels'] = test_dataloader.dataset.y
df.df['RealPredictions'] = predictions
print(df.df.to_string())

testing = Series.abs(df.df['TrueLabels'] - df.df['RealPredictions'])
print('Output ' + str(Series.sum(testing)) + ' incorrect out of ' + str(len(testing)))
