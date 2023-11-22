#!/usr/bin/env python3
"""
Imports
"""
import pandas as pd

def preprocess_data(file_path):
    """
    Preprocess Data
    """
    data = pd.read_csv(file_path)

    df = data.dropna()
    df = df[0::60]

    df['Timestamp'] = pd.to_datetime(df['Timestamp'],
                                     unit='s')
    df.reset_index(inplace=True,
                   drop=True)

    df = df[df['Timestamp'].dt.year >= 2017]
    df.reset_index(inplace=True,
                   drop=True)

    date_time = pd.to_datetime(df.pop('Timestamp'),
                               format='%Y-%m-%d %H:%M:%S')
    plot_features = df['Close']
    plot_features.index = date_time
    _ = plot_features.plot(subplots=True)

    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    train_df.to_csv('train_data.csv', index=False)
    val_df.to_csv('val_data.csv', index=False)
    test_df.to_csv('test_data.csv', index=False)

    df.to_csv('preprocessed_data.csv',
              index=False)

    return train_df, val_df, test_df

if __name__ == "__main__":
    file_path = "bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv"
    preprocess_data(file_path)
