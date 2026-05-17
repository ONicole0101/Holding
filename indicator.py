import pandas as pd

def add_indicators(df):

    low_min = df['low'].rolling(9).min()
    high_max = df['high'].rolling(9).max()

    rsv = (df['close'] - low_min) / (high_max - low_min) * 100

    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()

    df['MA'] = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()

    df['BB_upper'] = df['MA'] + 2 * std
    df['BB_lower'] = df['MA'] - 2 * std

    return df
