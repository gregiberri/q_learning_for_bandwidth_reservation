import numpy as np
import pandas as pd

NO1_path = 'datasets/Dataset_NO1.csv'

PRICE_RESOLUTION = 30


def load_and_preproces_datasets():
    no1_df = pd.read_csv(NO1_path)

    no1_df = no1_df[no1_df[' Price '] < 1]

    no1 = no1_df[no1_df[' Region'] == 'us-west-1b'].drop(' Region', axis=1)
    no2 = no1_df[no1_df[' Region'] == 'us-west-1c'].drop(' Region', axis=1)

    no1 = no1.drop(' Instance Type', axis=1)
    no2 = no2.drop(' Instance Type', axis=1)

    no1.Date = pd.to_datetime(no1.Date)
    no2.Date = pd.to_datetime(no2.Date)

    no1 = no1.set_index('Date')
    no2 = no2.set_index('Date')

    no1.columns = ['no1']
    no2.columns = ['no2']

    no1 = no1.resample(f'{PRICE_RESOLUTION}S').last().ffill()
    no2 = no2.resample(f'{PRICE_RESOLUTION}S').last().ffill()
    no_df = no1.merge(no2, on='Date', how='inner')

    # price scale to price resolution
    no_df /= 60 / PRICE_RESOLUTION
    no_df = no_df.reset_index()

    # price scale to 0-1
    min_price_at_resolution = np.min(no_df[['no1', 'no2']].to_numpy())
    max_price_at_resolution = np.max(no_df[['no1', 'no2']].to_numpy())
    no_df[['no1', 'no2']] = (no_df[['no1', 'no2']] - min_price_at_resolution) / (max_price_at_resolution - min_price_at_resolution)

    return no_df


def train_val_split(no_df: pd.DataFrame, train_val_rate: float = 0.7):
    train_val_split_index = int(train_val_rate * len(no_df))
    return no_df.loc[:train_val_split_index], no_df.loc[train_val_split_index:]
