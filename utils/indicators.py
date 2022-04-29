import pandas as pd
import numpy as np


def EMA(dataframe, column, column_name_output, period, alpha=False):
    # dataframe = dataframe.copy()
    con = pd.concat([dataframe[:period][column].rolling(window=period).mean(), dataframe[period:][column]])
    if (alpha == True):
        dataframe[column_name_output] = con.ewm(alpha=1 / period, adjust=False).mean()
    else:
        dataframe[column_name_output] = con.ewm(span=period, adjust=False).mean()

    dataframe[column_name_output].fillna(0, inplace=True)
    return dataframe


def ATR(dataframe, period, ohlc=['Open', 'High', 'Low', 'Close']):
    # dataframe = dataframe.copy()
    atr = 'ATR_' + str(period)

    if not 'TR' in dataframe.columns:
        dataframe['H-L'] = dataframe[ohlc[1]] - dataframe[ohlc[2]]
        dataframe['H-YC'] = abs(dataframe[ohlc[1]] - dataframe[ohlc[3]].shift())
        dataframe['L-YC'] = abs(dataframe[ohlc[2]] - dataframe[ohlc[3]].shift())

        dataframe['TR'] = dataframe[['H-L', 'H-YC', 'L-YC']].max(axis=1)

        dataframe.drop(['H-L', 'H-YC', 'L-YC'], inplace=True, axis=1)

    EMA(dataframe, 'TR', atr, period, alpha=False)

    return dataframe


def SuperTrend(dataframe, period=30, multiplier=3, ohlc=['Open', 'High', 'Low', 'Close']):
    dataframe = dataframe.copy()
    ATR(dataframe, period, ohlc=ohlc)
    atr = 'ATR_' + str(period)
    st = 'ST'
    stx = 'Label'

    dataframe['BasicUpperBand'] = (dataframe[ohlc[1]] + dataframe[ohlc[2]]) / 2 + multiplier * dataframe[atr]
    dataframe['BasicLowerBand'] = (dataframe[ohlc[1]] + dataframe[ohlc[2]]) / 2 - multiplier * dataframe[atr]

    dataframe['FinalUpperBand'] = 0.00
    dataframe['FinalLowerBand'] = 0.00
    for i in range(period, len(dataframe)):
        dataframe['FinalUpperBand'].iat[i] = dataframe['BasicUpperBand'].iat[i] if dataframe['BasicUpperBand'].iat[i] < \
                                                                                   dataframe['FinalUpperBand'].iat[
                                                                                       i - 1] or \
                                                                                   dataframe[ohlc[3]].iat[i - 1] > \
                                                                                   dataframe['FinalUpperBand'].iat[
                                                                                       i - 1] else \
            dataframe['FinalUpperBand'].iat[i - 1]
        dataframe['FinalLowerBand'].iat[i] = dataframe['BasicLowerBand'].iat[i] if dataframe['BasicLowerBand'].iat[i] > \
                                                                                   dataframe['FinalLowerBand'].iat[
                                                                                       i - 1] or \
                                                                                   dataframe[ohlc[3]].iat[i - 1] < \
                                                                                   dataframe['FinalLowerBand'].iat[
                                                                                       i - 1] else \
            dataframe['FinalLowerBand'].iat[i - 1]

    dataframe[st] = 0.00
    for i in range(period, len(dataframe)):
        dataframe[st].iat[i] = dataframe['FinalUpperBand'].iat[i] if dataframe[st].iat[i - 1] == \
                                                                     dataframe['FinalUpperBand'].iat[
                                                                         i - 1] and dataframe[ohlc[3]].iat[
                                                                         i] <= dataframe['FinalUpperBand'].iat[i] else \
            dataframe['FinalLowerBand'].iat[i] if dataframe[st].iat[i - 1] == dataframe['FinalUpperBand'].iat[i - 1] and \
                                                  dataframe[ohlc[3]].iat[i] > \
                                                  dataframe['FinalUpperBand'].iat[i] else \
                dataframe['FinalLowerBand'].iat[i] if dataframe[st].iat[i - 1] == dataframe['FinalLowerBand'].iat[
                    i - 1] and \
                                                      dataframe[ohlc[3]].iat[i] >= \
                                                      dataframe['FinalLowerBand'].iat[i] else \
                    dataframe['FinalUpperBand'].iat[i] if dataframe[st].iat[i - 1] == dataframe['FinalLowerBand'].iat[
                        i - 1] and \
                                                          dataframe[ohlc[3]].iat[i] < \
                                                          dataframe['FinalLowerBand'].iat[i] else 0.00

    dataframe[stx] = np.where((dataframe[st] > 0.00), np.where((dataframe[ohlc[3]] < dataframe[st]), 'DOWN', 'UP'),
                              np.NaN)

    dataframe.drop(['BasicUpperBand', 'BasicLowerBand', 'FinalUpperBand', 'FinalLowerBand'], inplace=True, axis=1)

    dataframe.fillna(0, inplace=True)
    return dataframe
