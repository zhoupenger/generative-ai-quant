import talib
import pandas as pd

def compute_indicators(df):
    df['SMA'] = talib.SMA(df['close'], timeperiod = 20)
    df['EMA'] = talib.EMA(df['close'], timeperiod = 20)
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['DX'] = talib.DX(df['high'], df['low'], df['close'], timeperiod=14)
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['BIAS'] = (df['close'] - df['SMA']) / df['SMA']
    df['ROC'] = talib.ROC(df['close'], timeperiod=10)

    df = df.dropna()
    df.reset_index(drop=True, inplace=True)

    return df