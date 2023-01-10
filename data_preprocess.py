import pandas as pd

NO1_path = 'datasets/Dataset_NO1.csv'
NO2_path = 'datasets/Dataset_NO2.csv'
NO3_path = 'datasets/Dataset_NO3.csv'

PRICE_RESOLUTION = 30
MAX_PRICE = 0.25


def load_and_preproces_datasets():
    no1_df = pd.read_csv(NO1_path)
    no2_df = pd.read_csv(NO2_path)
    no3_df = pd.read_csv(NO3_path)

    no1_df = no1_df[no1_df[' Region'] == 'us-west-1b'].drop(' Region', axis=1)
    no2_df = no2_df[no2_df[' Region'] == 'us-west-1b'].drop(' Region', axis=1)
    no3_df = no3_df[no3_df[' Region'] == 'us-west-1b'].drop(' Region', axis=1)

    no1_df = no1_df[no1_df[' Instance Type'] == 'c3.2xlarge'].drop(' Instance Type', axis=1)
    no2_df = no2_df[no2_df[' Instance Type'] == 'c3.2xlarge'].drop(' Instance Type', axis=1)
    no3_df = no3_df[no3_df[' Instance Type'] == 'c3.2xlarge'].drop(' Instance Type', axis=1)

    no1_df.Date = pd.to_datetime(no1_df.Date)
    no2_df.Date = pd.to_datetime(no2_df.Date)
    no3_df.Date = pd.to_datetime(no3_df.Date)

    no1 = no1_df.set_index('Date')
    no2 = no2_df.set_index('Date')
    no3 = no3_df.set_index('Date')

    no1.columns = ['no1']
    no2.columns = ['no2']
    no3.columns = ['no3']

    no1 = no1.resample(f'{PRICE_RESOLUTION}S').last().ffill()
    no2 = no2.resample(f'{PRICE_RESOLUTION}S').last().ffill()
    no3 = no3.resample(f'{PRICE_RESOLUTION}S').last().ffill()
    no_df = no1.merge(no2, on='Date', how='inner').merge(no3, on='Date', how='inner')

    # price scale to 0-1
    no_df[['no1', 'no2', 'no3']] /= MAX_PRICE

    # price scale to price resolution
    no_df /= 60 / PRICE_RESOLUTION
    no_df = no_df.reset_index()

    return no_df


def train_val_split(no_df: pd.DataFrame, train_val_rate: float = 0.7):
    train_val_split_index = int(train_val_rate * len(no_df))
    return no_df.loc[:train_val_split_index], no_df.loc[train_val_split_index:]
