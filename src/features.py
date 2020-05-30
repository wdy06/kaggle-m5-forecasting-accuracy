import numpy as np
import pandas as pd
import sandesh
from tqdm import tqdm
import traceback

import preprocessing
import utils
from dataset import M5Dataset
from create_folds import create_folds


def simple_feature(data):
    id_columns = ['all_id', 'item_id', 'dept_id',
                  'cat_id', 'store_id', 'state_id', 'date']
    ret_df = data[id_columns].copy()

    group_ids = ('all_id', 'state_id', 'store_id', 'cat_id', 'dept_id', 'item_id',
                 ['state_id', 'cat_id'],  ['state_id',
                                           'dept_id'], ['store_id', 'cat_id'],
                 ['store_id', 'dept_id'], ['item_id', 'state_id'], ['item_id', 'store_id'])
    group_id_names = ('all', 'state', 'store', 'cat', 'dept', 'item',
                      'state_cat', 'state_dept', 'store_cat',
                      'store_dept', 'item_state', 'item_store')
    # data['all_id'] = 'all'
    # rolling demand features
    # lag feature
    # lag_list = [28, 29, 30]
    # -------------------------------
    for lag in tqdm([28, 29, 30]):
        ret_df[f'lag_t{lag}'] = data.groupby(
            ['id'])['demand'].transform(lambda x: x.shift(lag))

    for lag in tqdm([28, 29, 30]):
        for group_id, id_name in zip(group_ids, group_id_names):
            if isinstance(group_id, str):
                group_id = [group_id]
            tmp = data.groupby(
                group_id + ['date'])['demand'].mean().groupby(group_id).apply(lambda x: x.shift(lag))
            tmp.name = f'lag_t{lag}_{id_name}'
            tmp = tmp.reset_index()
            ret_df = pd.merge(ret_df, tmp, on=group_id + ['date'])
    # -------------------------------

    # rolling mean
    # rolling_list = [7, 14, 21, 28]
    # for window_size in rolling_list:
    #     data[f'rolling_mean_t{window_size}'] = data.groupby(
    #         ['id'])['demand'].transform(lambda x: x.rolling(window_size).mean())

    # lag rolling
    for lag in [28]:
        for window in tqdm([7, 30, 90, 180]):

            ret_df[f'lag_t{lag}_rolling_mean_t{window}'] = data.groupby(
                ['id'])['demand'].transform(lambda x: x.shift(lag).rolling(window).mean())

            for group_id, id_name in zip(group_ids, group_id_names):
                if isinstance(group_id, str):
                    group_id = [group_id]
                agg_dict = {f'lag_t{lag}_rolling_mean_t{window}_{id_name}': 'mean',
                            f'lag_t{lag}_rolling_std_t{window}_{id_name}': 'std'}
                tmp = data.groupby(
                    group_id + ['date'])['demand'].mean().groupby(group_id).apply(lambda x: x.shift(lag).rolling(window).agg(agg_dict))
                # tmp.name = f'lag_t{lag}_rolling_mean_t{window}_{id_name}'
                tmp = tmp.reset_index()
                ret_df = pd.merge(ret_df, tmp, on=group_id + ['date'])

    ret_df = ret_df.drop(id_columns, axis=1)
    return ret_df


def price_feature(data):
    ret_df = pd.DataFrame()
    ret_df['lag_price_t1'] = data.groupby(
        ['id'])['sell_price'].transform(lambda x: x.shift(1))
    ret_df['price_change_t1'] = (
        ret_df['lag_price_t1'] - data['sell_price']) / (ret_df['lag_price_t1'])
    ret_df['rolling_price_max_t365'] = data.groupby(
        ['id'])['sell_price'].transform(lambda x: x.shift(1).rolling(365).max())
    ret_df['price_change_t365'] = (
        ret_df['rolling_price_max_t365'] - data['sell_price']) / (ret_df['rolling_price_max_t365'])
    ret_df['rolling_price_std_t7'] = data.groupby(
        ['id'])['sell_price'].transform(lambda x: x.rolling(7).std())
    ret_df['rolling_price_std_t30'] = data.groupby(
        ['id'])['sell_price'].transform(lambda x: x.rolling(30).std())
    ret_df.drop(['rolling_price_max_t365', 'lag_price_t1'],
                inplace=True, axis=1)

    return ret_df


def date_feature(data):
    ret_df = pd.DataFrame()
    # data['date'] = pd.to_datetime(data['date'])
    ret_df['year'] = data['date'].dt.year
    ret_df['month'] = data['date'].dt.month
    ret_df['week'] = data['date'].dt.week
    ret_df['day'] = data['date'].dt.day
    ret_df['dayofweek'] = data['date'].dt.dayofweek
    return ret_df


def generate_features(data, save=False):
    ret_df = simple_feature(data)
    if save:
        utils.dump_pickle(ret_df, utils.FEATURE_DIR / 'simple_feature.pkl')
    tmp = price_feature(data)
    if save:
        utils.dump_pickle(tmp, utils.FEATURE_DIR / 'price_feature.pkl')
    ret_df = pd.concat([ret_df, tmp], axis=1)
    tmp = date_feature(data)
    if save:
        utils.dump_pickle(tmp, utils.FEATURE_DIR / 'date_feature.pkl')
    ret_df = pd.concat([ret_df, tmp], axis=1)
    return ret_df


if __name__ == '__main__':
    output_path = utils.FEATURE_DIR / 'baseline_features.pkl'
    encoder_path = utils.FEATURE_DIR / 'encoder.pkl'
    melted_path = utils.FEATURE_DIR / 'melted.pkl'
    fold_indices_path = utils.FEATURE_DIR / 'fold_indices.pkl'
    sandesh.send(f'start generating feature')
    try:
        print('generating features...')
        # dataset = M5Dataset()
        # data = preprocessing.melt_and_merge(
        #     dataset.calendar, dataset.sell_prices, dataset.main_df, dataset.submission,
        #     merge=True)
        # data = preprocessing.add_separated_item_id(data)
        # utils.dump_pickle(data, melted_path)
        data = utils.load_pickle(melted_path)
        data['date'] = pd.to_datetime(data['date'])
        data['all_id'] = 'all'
        # label encoding
        cat = ['item_id', 'item_id_1', 'item_id_2', 'item_id_3', 'dept_id', 'cat_id', 'store_id', 'state_id',
               'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
        data, encoder = preprocessing.label_encoding(df=data, cat_features=cat,
                                                     verbose=True)
        tmp = generate_features(data, save=True)
        data = pd.concat([data, tmp], axis=1)
        fold_indices = create_folds(data)
        utils.dump_pickle(fold_indices, fold_indices_path)
        utils.dump_pickle(data, output_path)
        utils.dump_pickle(encoder, encoder_path)
        print('finished generating features !!')
    except Exception as e:
        sandesh.send(str(e))
        raise
    else:
        sandesh.send(f'finished generating features !!')
