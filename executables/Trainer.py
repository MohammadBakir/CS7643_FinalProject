import warnings

from pandas import Series

from models import TransformerModelImpl
from models.FullyConnectedNetwork import FCNet
from models.LSTM import LSTM
from models.LSTM_CNN import LSTM_CNN
from models.my_transformer import TransformerModelImpl2
import torchvision.models as tmodels
import torch.nn.functional as f

warnings.simplefilter('ignore')
import torch
from torch.utils.data import DataLoader

from utils.get_dataset import StockData, GetDataset
from utils.utils import train, evaluate, plot_curves

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HYPERPARAMETERS:
    NUM_DAYS = 5
    BATCH_SIZE = 8
    DEVICE = device
    TMI_N_LAYERS = 4
    TMI_NUM_HEADS = 2
    FEATURES = 13
    TMI_FORWARD_DIM = 8
    TMI_OUTPUT_DIM = 6
    OUTPUT_DIM = 1
    DROPOUT = 0.1
    LR = 0.001
    SEQ_LENGTH = 5


class params:
    device = device
    num_layers = 1
    nhead = 1
    d_model = 9
    dim_feedforward = 128
    d_output = 1
    dropout = 3e-4
    seq_len = 7


epochs = 200
image_name = 'RNN-CNN SPY'
model_save_path_and_name = "../outputs/RNN-CNN_SPY_ST_Python_Testing-test.pth"
save_model = True
overlap_training = True
overlap_validation = True
overlap_testing = True
shuffle_training = True
shuffle_validation = True
shuffle_testing = False

# MODEL = TransformerModelImpl(HYPERPARAMETERS).to(device)
# #MODEL = TransformerModelImpl2(params).to(device)
# MODEL = FCNet(in_shape=HYPERPARAMETERS.FEATURES * HYPERPARAMETERS.NUM_DAYS)
# Model Types can be 'rnn', 'lstm', and 'gru'
# MODEL = LSTM(modeltype='rnn', input_size=10, lstm_hidden_size=5, lstm_layers=5, lstm_output_size=1, leaky_relu=0.2)
MODEL = LSTM_CNN(modeltype='rnn', input_size=12, lstm_hidden_size=5, lstm_layers=5, lstm_output_size=1, kernel_size=3,
                 padding=1)
# MODEL = FullyConnectedNetwork

CRITERION = torch.nn.BCEWithLogitsLoss(reduction='mean')
OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=HYPERPARAMETERS.LR)

csv = '../data/SPY-Indicators.csv'
df = GetDataset(csv)
dataset = df.get_data()
valid_frac, test_frac = 0.2, 0.2
train_sz = int(dataset.shape[0] * (1 - (valid_frac + test_frac)))
valid_sz = int(dataset.shape[0] * (valid_frac))
df_train = dataset[0:train_sz]
df_valid = dataset[train_sz:train_sz + valid_sz]
df_test = dataset[train_sz + valid_sz:]
train_dataset = StockData(df_train.to_numpy(), num_days=HYPERPARAMETERS.NUM_DAYS, overlap=overlap_training)
valid_dataset = StockData(df_valid.to_numpy(), num_days=HYPERPARAMETERS.NUM_DAYS, overlap=overlap_validation)
test_dataset = StockData(df_test.to_numpy(), num_days=HYPERPARAMETERS.NUM_DAYS, overlap=overlap_testing)
train_loader = DataLoader(train_dataset,
                          batch_size=HYPERPARAMETERS.BATCH_SIZE,
                          shuffle=shuffle_training)
valid_loader = DataLoader(valid_dataset,
                          batch_size=HYPERPARAMETERS.BATCH_SIZE,
                          shuffle=shuffle_validation)
test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=shuffle_testing)

avg_train_loss, avg_train_acc, avg_valid_loss, avg_valid_acc = [], [], [], []
MODEL.float()
MODEL.to(device)
for epoch in range(epochs):
    train_loss, atl, ata = train(MODEL, train_loader, OPTIMIZER, CRITERION, device)
    # scheduler.step(train_loss)
    _, avl, ava = evaluate(MODEL, valid_loader, CRITERION, device)
    # if epoch % 50 == 1:
    print(
        "Epoch %d: Training Loss: %.4f. Training Accuracy: %.4f. Validation Loss: %.4f. Validation Accuracy: %.4f." % (
            epoch + 1, atl, ata, avl, ava))
    avg_train_loss.append(atl.item())
    avg_train_acc.append(ata)
    avg_valid_loss.append(avl.item())
    avg_valid_acc.append(ava)

plot_curves(avg_train_loss, avg_train_acc, avg_valid_loss, avg_valid_acc, info=image_name, save=True,
            path='../outputs/')

a, b = train_dataset[:]
# print(a.shape)
# print(b.shape)
# print(b)

from sklearn.metrics import classification_report, accuracy_score

features, targets = test_dataset[:]
features, targets = features.to(device), targets.to(device)
MODEL.eval()
with torch.no_grad():
    predictions = MODEL(features.float())
    # Added back the rounding and sigmoid in training, validation, and testing in Trainer.py
    # Reference: https://stackoverflow.com/questions/64002566/bcewithlogitsloss-trying-to-get-binary-output-for-predicted-label-as-a-tensor
    pred = torch.round(f.sigmoid(predictions)).long()
    print(classification_report(targets.cpu().detach().numpy(), pred.cpu().detach().numpy()))
testing_results_dataframe = df.raw_df.tail(len(pred))
testing_results_dataframe['ModelPredictions'] = pred
print(testing_results_dataframe.to_string())

testing = Series.abs(testing_results_dataframe['Next_Day_Change'] - testing_results_dataframe['ModelPredictions'])
print('Output ' + str(Series.sum(testing)) + ' incorrect out of ' + str(len(testing)))

print('Testing Accuracy: ' + str(accuracy_score(targets, pred)))

if save_model:
    torch.save(MODEL.state_dict(), model_save_path_and_name)
