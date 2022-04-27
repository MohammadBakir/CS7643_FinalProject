import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch


class StockData(Dataset):

    def __init__(self, data, num_days=2):
        x_temp = data[:, :-1]
        y_temp = data[:, -1]

        shape = (x_temp.shape[0] - num_days + 1, num_days, x_temp.shape[1])
        strides = (x_temp.strides[0], x_temp.strides[0], x_temp.strides[1])

        self.features = np.lib.stride_tricks.as_strided(x_temp, shape, strides)
        self.x = torch.from_numpy(np.lib.stride_tricks.as_strided(x_temp, shape, strides))
        self.labels = y_temp[num_days - 1:].reshape(-1, 1).astype('int')
        self.y = torch.from_numpy(y_temp[num_days - 1:].reshape(-1, 1).astype('int'))

        self.num_samples = self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.num_samples


class GetDataset(object):
    def __init__(self, csv):
        super(GetDataset, self).__init__()
        self.csv = csv

    def get_data(self):
        '''Get Data into Dataframe'''
        self.df = pd.read_csv(self.csv)
        self.df.drop('Symbol', 1, inplace=True)
        self.df['DateTime'] = pd.to_datetime(self.df['DateTime']).dt.date
        self.df.rename({'DateTime': 'Date'}, axis=1, inplace=True)

        '''Generate Classification Column'''
        # note ilkay made some fixes
        self.df['Next_Close_Day'] = self.df['Close'].shift(-1)
        self.df.dropna(how='any', axis=0, inplace=True)  # if this is not done if/else logic below will fill NaN with 0
        comparison_column = np.where(self.df["Next_Close_Day"] > self.df["Close"], int(1), int(0))
        self.df["change"] = comparison_column
        self.df['Next_Day_Change'] = self.df['change']
        self.df.drop(["Next_Close_Day", "change"], axis=1, inplace=True)

        '''Indicators'''
        self.df['EMA'] = np.where(self.df.Close > self.df.EMA, 1.0, 0.0)
        self.df['SuperTrend1'] = np.where(self.df.Close > self.df.SuperTrend1, 1.0, 0.0)
        self.df['SuperTrend2'] = np.where(self.df.Close > self.df.SuperTrend2, 1.0, 0.0)
        self.df['SuperTrend3'] = np.where(self.df.Close > self.df.SuperTrend3, 1.0, 0.0)
        self.df['SuperTrend4'] = np.where(self.df.Close > self.df.SuperTrend4, 1.0, 0.0)

        '''Calculate percentage change'''
        self.df['Open'] = self.df['Open'].pct_change()  # Create arithmetic returns column
        self.df['High'] = self.df['High'].pct_change()  # Create arithmetic returns column
        self.df['Low'] = self.df['Low'].pct_change()  # Create arithmetic returns column
        self.df['Close'] = self.df['Close'].pct_change()  # Create arithmetic returns column
        self.df['Volume'] = self.df['Volume'].pct_change()

        self.df.dropna(how='any', axis=0, inplace=True)  # Drop all rows with NaN values

        # todo implement standard scaler instead, also need to move scaling to dataset's transform method
        '''Normalize price columns'''
        self.df[["Open", "High", "Low", "Close", "Volume"]] = self.df[["Open", "High", "Low", "Close", "Volume"]].apply(
            self.normalize_data)
        # self.df[["Open", "High", "Low", "Close", "EMA"]] = self.df[["Open", "High", "Low", "Close", "EMA"]].apply(self.normalize_data)

        '''Drop Date Column'''
        self.df.drop(columns=['Date'], inplace=True)

        return self.df

    def get_data2(self, future_days):
        '''Get Data into Dataframe'''
        self.cols = pd.read_csv(self.csv, nrows=0).columns.tolist()
        self.data = pd.read_csv(self.csv, sep=',', names=self.cols, skiprows=[0])
        self.df = self.data.iloc[:, 0].str.split(',', expand=True)
        self.cols = self.cols[0].split(',')
        self.df.columns = self.cols
        self.df.drop('Symbol', 1, inplace=True)
        self.df[["Open", "High", "Low", "Close"]] = self.df[["Open", "High", "Low", "Close"]].apply(pd.to_numeric)
        self.df['DateTime'] = pd.to_datetime(self.df['DateTime']).dt.date
        self.df.rename({'DateTime': 'Date'}, axis=1, inplace=True)

        '''Generate Classification Column'''
        # note ilkay made some fixes
        self.df['Next_Close_Day'] = self.df['Close'].shift(-1 * future_days)
        self.df.dropna(how='any', axis=0, inplace=True)  # if this is not done if/else logic below will fill NaN with 0
        comparison_column = np.where(self.df["Next_Close_Day"] > self.df["Close"], int(1), int(0))
        self.df["change"] = comparison_column
        self.df['Next_Day_Change'] = self.df['change']
        self.df.drop(["Next_Close_Day", "change"], axis=1, inplace=True)

        '''Calculate percentage change'''
        self.df['ema10'] = self.df['Close'].ewm(span=10).mean()
        self.df['ema20'] = self.df['Close'].ewm(span=20).mean()
        self.df['ma10'] = self.df['Close'].rolling(10).mean()
        self.df['ma20'] = self.df['Close'].rolling(20).mean()
        self.df['ma50'] = self.df['Close'].rolling(50).mean()

        self.df['ema10'] /= self.df['Close']
        self.df['ema20'] /= self.df['Close']
        self.df['ma10'] /= self.df['Close']
        self.df['ma20'] /= self.df['Close']
        self.df['ma50'] /= self.df['Close']

        self.df['ema10'] -= 1
        self.df['ema20'] -= 1
        self.df['ma10'] -= 1
        self.df['ma20'] -= 1
        self.df['ma50'] -= 1

        self.df.dropna(how='any', axis=0, inplace=True)  # Drop all rows with NaN values

        '''Drop Date Column'''
        self.df.drop(columns=['Date', 'Close', 'Open', 'Low', 'High'], inplace=True)

        '''Normalize'''
        self.df[['ema10', 'ema20', 'ma10', 'ma20', 'ma50']] = self.df[['ema10', 'ema20', 'ma10', 'ma20', 'ma50']].apply(
            self.normalize_data)

        return self.df

    def normalize_data(self, df):
        min = df.min()
        max = df.max()
        x = df
        y = (x - min) / (max - min)

        return y

    def generate_windows(self, train_split_ratio=0.80, val_split_ratio=.10, time_period=2):
        '''Seperate data into Features/Target'''
        x_data = self.df.iloc[:, :-1].shift(-1)
        y_data = self.df.iloc[:, -1]

        '''Generate Data Splits'''
        train_data_size = int(np.ceil(len(self.df) * train_split_ratio))
        x_train_data = x_data[:train_data_size]
        y_train_data = y_data[:train_data_size]

        self.x_train = [x_train_data[i - time_period:i] for i in range(time_period, len(x_train_data), time_period)]
        self.y_train = y_train_data[time_period:]

        x_remain_data = x_data[train_data_size - time_period:]
        y_remain = y_data[train_data_size:]
        x_remain = [x_remain_data[i - time_period:i] for i in range(time_period, len(x_remain_data))]

        val_data_size = int(np.ceil(len(x_data) * val_split_ratio))

        self.x_val = x_remain[0:val_data_size + 1]
        self.y_val = y_remain[0:val_data_size + 1]

        self.x_test = x_remain[val_data_size + 1::]
        self.y_test = y_remain[val_data_size + 1::]

        self.y_train = np.array(self.y_train)
        self.x_train = np.array(self.x_train)

        print(f'Shape of train data: (x, y) = ({np.shape(self.x_train)}, {np.shape(self.y_train)})')

        self.x_val = np.array(self.x_val)
        self.y_val = np.array(self.y_val)

        print(f'Shape of val data: (x, y) = ({np.shape(self.x_val)}, {np.shape(self.y_val)})')

        self.x_test = np.array(self.x_test)
        self.y_test = np.array(self.y_test)

        print(f'Shape of test data: (x, y) = ({np.shape(self.x_test)}, {np.shape(self.y_test)})')

        return [self.x_train, self.y_train.astype('int')], [self.x_val, self.y_val.astype('int')], [self.x_test,
                                                                                                    self.y_test.astype(
                                                                                                        'int')]
