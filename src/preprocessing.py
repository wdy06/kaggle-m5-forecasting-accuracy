import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed

import utils
from tqdm import tqdm


def melt_and_merge(calendar, sell_prices, sales_train_validation, submission,
                   merge=True, fill_na=True):

    # melt sales data, get it ready for training
    sales_train_validation = pd.melt(sales_train_validation, id_vars=[
                                     'id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name='day', value_name='demand')
    print('Melted sales train validation has {} rows and {} columns'.format(
        sales_train_validation.shape[0], sales_train_validation.shape[1]))
    # sales_train_validation = utils.reduce_mem_usage(sales_train_validation)

    # seperate test dataframes
    test1_rows = [row for row in submission['id'] if 'validation' in row]
    test2_rows = [row for row in submission['id'] if 'evaluation' in row]
    test1 = submission[submission['id'].isin(test1_rows)]
    test2 = submission[submission['id'].isin(test2_rows)]

    # change column names
    test1.columns = ['id', 'd_1914', 'd_1915', 'd_1916', 'd_1917', 'd_1918', 'd_1919', 'd_1920', 'd_1921', 'd_1922', 'd_1923', 'd_1924', 'd_1925', 'd_1926', 'd_1927', 'd_1928', 'd_1929', 'd_1930', 'd_1931',
                     'd_1932', 'd_1933', 'd_1934', 'd_1935', 'd_1936', 'd_1937', 'd_1938', 'd_1939', 'd_1940', 'd_1941']
    test2.columns = ['id', 'd_1942', 'd_1943', 'd_1944', 'd_1945', 'd_1946', 'd_1947', 'd_1948', 'd_1949', 'd_1950', 'd_1951', 'd_1952', 'd_1953', 'd_1954', 'd_1955', 'd_1956', 'd_1957', 'd_1958', 'd_1959',
                     'd_1960', 'd_1961', 'd_1962', 'd_1963', 'd_1964', 'd_1965', 'd_1966', 'd_1967', 'd_1968', 'd_1969']

    # get product table
    product = sales_train_validation[[
        'id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()

    # merge with product table
    test2['id'] = test2['id'].str.replace('_evaluation', '_validation')
    test1 = test1.merge(product, how='left', on='id')
    test2 = test2.merge(product, how='left', on='id')
    test2['id'] = test2['id'].str.replace('_validation', '_evaluation')

    #
    test1 = pd.melt(test1, id_vars=['id', 'item_id', 'dept_id', 'cat_id',
                                    'store_id', 'state_id'], var_name='day', value_name='demand')
    test2 = pd.melt(test2, id_vars=['id', 'item_id', 'dept_id', 'cat_id',
                                    'store_id', 'state_id'], var_name='day', value_name='demand')

    sales_train_validation['part'] = 'train'
    test1['part'] = 'test1'
    test2['part'] = 'test2'

    data = pd.concat([sales_train_validation, test1, test2], axis=0)

    # del sales_train_validation, test1, test2

    # get only a sample for fst training
    # data = data.loc[nrows:]

    # drop some calendar features
    calendar.drop(['weekday', 'wday', 'month', 'year'], inplace=True, axis=1)

    # delete test2 for now
    # TODO: update for test2
    data = data[data['part'] != 'test1']

    if merge:
        # notebook crash with the entire dataset (maybee use tensorflow, dask, pyspark xD)
        data = pd.merge(data, calendar, how='left',
                        left_on=['day'], right_on=['d'])
        data.drop(['d', 'day'], inplace=True, axis=1)
        # get the sell price data (this feature should be very important)
        data = data.merge(
            sell_prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
        print('Our final dataset to train has {} rows and {} columns'.format(
            data.shape[0], data.shape[1]))

    if fill_na:
        nan_features = ['event_name_1', 'event_type_1',
                        'event_name_2', 'event_type_2']
        for feature in nan_features:
            data[feature].fillna('unknown', inplace=True)

    # if label_encoding:
    #     cat = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
    #            'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    #     for feature in cat:
    #         encoder = LabelEncoder()
    #         data[feature] = encoder.fit_transform(data[feature])
    return data


def melt_to_pivot(melt_df, target_col, encoder_dict):
    index_columns = ['id', 'item_id', 'dept_id',
                     'cat_id', 'store_id', 'state_id']
    index_df = melt_df[index_columns].drop_duplicates().copy()
    demand_df = pd.pivot(melt_df, index='id', columns='date',
                         values=target_col).reset_index().copy()
    demand_df = pd.merge(index_df, demand_df, on='id', how='left')
    demand_df = demand_df.reset_index(drop=True)
    # print(demand_df)
    # print(demand_df.columns)
    demand_df.columns = index_columns + \
        [f'd_{i}' for i in range(
            1, len(demand_df.columns) - len(index_columns) + 1)]

    # print('decoding categorical columns...')
    for col in index_columns:
        # print(col)
        encoder = encoder_dict.get(col)
        if encoder is not None:
            # print(f'aa: {col}')
            demand_df[col] = encoder.inverse_transform(demand_df[col])
    return demand_df


def label_encoding(df, cat_features, verbose=False):
    encoder_dict = {}
    for feature in cat_features:
        if verbose:
            print(f'label encoding {feature} ...')
        encoder = LabelEncoder()
        df[feature] = encoder.fit_transform(df[feature].astype(str))
        encoder_dict[feature] = encoder
    return df, encoder_dict


def add_separated_item_id(df):
    tmp = df['item_id'].str.split('_', n=2, expand=True)
    tmp[0] = tmp[0].str.cat(tmp[1], sep='_')
    tmp[1] = tmp[2].str.slice(stop=1)
    tmp[2] = tmp[2].str.slice(start=1)
    tmp.columns = ['item_id_1', 'item_id_2', 'item_id_3']
    df = pd.concat([df, tmp], axis=1)
    return df
