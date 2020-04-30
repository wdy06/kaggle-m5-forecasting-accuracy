import numpy as np
import pandas as pd


def create_folds(X):
    print('creating folds ...')
    query_list = [
        ('date <= "2015-04-26"', '"2015-04-26" < date <= "2015-05-24"'),
        ('date <= "2015-05-24"', '"2015-05-24" < date <= "2015-06-21"'),
        ('date <= "2016-01-31"', '"2016-01-31" < date <= "2016-02-28"'),
        ('date <= "2016-02-28"', '"2016-02-28" < date <= "2016-03-27"'),
        ('date <= "2016-03-27"', '"2016-03-27" < date <= "2016-04-24"')
    ]
    fold_indices = []
    X.reset_index(inplace=True)
    for query in query_list:
        print(query)
        train_idx = X.query(query[0]).index.tolist()
        val_idx = X.query(query[1]).index.tolist()
        # all_val_idx = all_val_idx + val_idx
        fold_indices.append((train_idx, val_idx))
    return fold_indices
