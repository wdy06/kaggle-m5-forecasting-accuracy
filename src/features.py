import numpy as np
import pandas as pd

import preprocessing
import utils
from dataset import M5Dataset


def simple_feature(data):

    # rolling demand features
    # lag feature
    # lag_list = [28, 29, 30]
    for lag in [28, 29, 30]:
        data[f'lag_t{lag}'] = data.groupby(
            ['id'])['demand'].transform(lambda x: x.shift(lag))
    # data['lag_t7'] = data.groupby(
    #     ['id'])['demand'].transform(lambda x: x.shift(7))
    # data['lag_t28'] = data.groupby(
    #     ['id'])['demand'].transform(lambda x: x.shift(28))
    # data['lag_t29'] = data.groupby(
    #     ['id'])['demand'].transform(lambda x: x.shift(29))
    # data['lag_t30'] = data.groupby(
    #     ['id'])['demand'].transform(lambda x: x.shift(30))

    # rolling mean
    # rolling_list = [7, 14, 21, 28]
    # for window_size in rolling_list:
    #     data[f'rolling_mean_t{window_size}'] = data.groupby(
    #         ['id'])['demand'].transform(lambda x: x.rolling(window_size).mean())

    # lag rolling
    for lag in [28]:
        for window in [7, 30, 90, 180]:
            data[f'lag{lag}_rolling_mean_t{window}'] = data.groupby(
                ['id'])['demand'].transform(lambda x: x.shift(lag).rolling(window).mean())
            data[f'lag{lag}_rolling_std_t{window}'] = data.groupby(
                ['id'])['demand'].transform(lambda x: x.shift(lag).rolling(window).std())
            data[f'lag{lag}_rolling_skew_t{window}'] = data.groupby(
                ['id'])['demand'].transform(lambda x: x.shift(lag).rolling(window).skew())
            data[f'lag{lag}_rolling_kurt_t{window}'] = data.groupby(
                ['id'])['demand'].transform(lambda x: x.shift(lag).rolling(window).kurt())
    # data['lag28_rolling_mean_t7'] = data.groupby(['id'])['demand'].transform(
    #     lambda x: x.shift(28).rolling(7).mean())
    # data['lag28_rolling_std_t7'] = data.groupby(['id'])['demand'].transform(
    #     lambda x: x.shift(28).rolling(7).std())
    # data['lag28_rolling_mean_t30'] = data.groupby(['id'])['demand'].transform(
    #     lambda x: x.shift(28).rolling(30).mean())
    # data['lag28_rolling_mean_t90'] = data.groupby(['id'])['demand'].transform(
    #     lambda x: x.shift(28).rolling(90).mean())
    # data['lag28_rolling_mean_t180'] = data.groupby(['id'])['demand'].transform(
    #     lambda x: x.shift(28).rolling(180).mean())
    # data['lag28_rolling_std_t30'] = data.groupby(['id'])['demand'].transform(
    #     lambda x: x.shift(28).rolling(30).std())
    # data['lag28_rolling_skew_t30'] = data.groupby(['id'])['demand'].transform(
    #     lambda x: x.shift(28).rolling(30).skew())
    # data['lag28_rolling_kurt_t30'] = data.groupby(['id'])['demand'].transform(
    #     lambda x: x.shift(28).rolling(30).kurt())

    # price features
    data['lag_price_t1'] = data.groupby(
        ['id'])['sell_price'].transform(lambda x: x.shift(1))
    data['price_change_t1'] = (
        data['lag_price_t1'] - data['sell_price']) / (data['lag_price_t1'])
    data['rolling_price_max_t365'] = data.groupby(
        ['id'])['sell_price'].transform(lambda x: x.shift(1).rolling(365).max())
    data['price_change_t365'] = (
        data['rolling_price_max_t365'] - data['sell_price']) / (data['rolling_price_max_t365'])
    data['rolling_price_std_t7'] = data.groupby(
        ['id'])['sell_price'].transform(lambda x: x.rolling(7).std())
    data['rolling_price_std_t30'] = data.groupby(
        ['id'])['sell_price'].transform(lambda x: x.rolling(30).std())
    data.drop(['rolling_price_max_t365', 'lag_price_t1'], inplace=True, axis=1)

    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['week'] = data['date'].dt.week
    data['day'] = data['date'].dt.day
    data['dayofweek'] = data['date'].dt.dayofweek

    return data


def generate_features(data):
    data = simple_feature(data)
    return data


if __name__ == '__main__':
    output_path = utils.FEATURE_DIR / 'baseline_features.pkl'
    encoder_path = utils.FEATURE_DIR / 'encoder.pkl'
    print('generating features...')
    dataset = M5Dataset()
    data = preprocessing.melt_and_merge(
        dataset.calendar, dataset.sell_prices, dataset.main_df, dataset.submission,
        merge=True)
    # label encoding
    cat = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
           'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    data, encoder = preprocessing.label_encoding(df=data, cat_features=cat,
                                                 verbose=True)
    data = generate_features(data)
    utils.dump_pickle(data, output_path)
    utils.dump_pickle(encoder, encoder_path)
    print('finished generating features !!')
