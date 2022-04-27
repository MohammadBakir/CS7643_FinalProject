from data.get_dataset import GetDataset, StockData
from models.FCNet import FCNet
from models.my_transformer import TransformerModelImpl
from torch.utils.data import DataLoader
from torch import nn
from utils import plot_curves
from utils import train
from utils import evaluate
import torch
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


FAKE_DATA=False
REPEAT_ONE_SMALL_BATCH=False
USE_TRANSFORMER=False
NUM_DAYS=1


class hyperparameters:
    n_layers = 1
    num_heads = 1
    model_dim = 4
    forward_dim = 16
    output_dim = 1
    dropout = 3e-4
    n_epochs = 100
    lr = 0.1 if USE_TRANSFORMER else 0.01
    batch_size = 1024


csv = './data/SPXDailyData.csv'
df = GetDataset(csv)
dataset = df.get_data()
if REPEAT_ONE_SMALL_BATCH:
    dataset = dataset[0:10]
    if FAKE_DATA:
        dataset[:][0:5]=.6
        dataset[:][5:]=.4
        dataset['Next_Day_Change'][0:5]=0
        dataset['Next_Day_Change'][5:]=1

#split into 3
valid_frac, test_frac = 0.2, 0.2
train_sz=int(dataset.shape[0]*(1-(valid_frac+test_frac)))
valid_sz=int(dataset.shape[0]*(valid_frac))
df_train = dataset[               0:train_sz]
df_valid = dataset[        train_sz:train_sz+valid_sz]
df_test = dataset[train_sz+valid_sz:]

#convert to sequence data and make dataset
train_dataset = StockData(df_train.to_numpy(), num_days=NUM_DAYS)
valid_dataset = StockData(df_valid.to_numpy(), num_days=NUM_DAYS)
test_dataset = StockData(df_test.to_numpy(), num_days=NUM_DAYS)


modelT = TransformerModelImpl(hyperparameters)
train_loader = DataLoader(train_dataset, batch_size=train_dataset.num_samples if REPEAT_ONE_SMALL_BATCH else hyperparameters.batch_size, shuffle=False)#todo while debuggin set to false
valid_loader = DataLoader(valid_dataset, batch_size=valid_dataset.num_samples if REPEAT_ONE_SMALL_BATCH else hyperparameters.batch_size, shuffle=False)#todo while debuggin set to false

modelFC = FCNet((train_dataset[:][0].shape[2])*NUM_DAYS)

model = modelT if USE_TRANSFORMER else modelFC
criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters.lr)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=1)#constant
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

avg_train_loss,avg_train_acc,avg_valid_loss,avg_valid_acc=[],[],[],[]
model.float()
for epoch in range(hyperparameters.n_epochs):
    train_loss, atl, ata = train(model, train_loader, optimizer, criterion)
    scheduler.step(train_loss)
    _, avl, ava = evaluate(model, valid_loader, criterion)
    if epoch%50==1:
        print("Epoch %d: Training Loss: %.4f. Training Acc: %.4f. Validation Loss: %.4f. Validation Acc: %.4f." % (epoch+1, atl, ata, avl, ava))
    avg_train_loss.append(atl.item())
    avg_train_acc.append(ata)
    avg_valid_loss.append(avl.item())
    avg_valid_acc.append(ava)
plot_curves(avg_train_loss,avg_train_acc,avg_valid_loss,avg_valid_acc, info='', save=False)


model.eval()
features, targets = test_dataset[:]
predictions = model(features.float())
pred=torch.round(torch.sigmoid(predictions)).long()
print(classification_report(targets.detach().numpy(), pred.detach().numpy(), output_dict=True)['weighted avg']['f1-score'])