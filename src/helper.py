from tqdm import tqdm
import pandas as pd

import features


def recursive_predict(runner, feature_config, all_features, X_all, X_test, cv=False):
    test_dates = ['2016-04-25', '2016-04-26', '2016-04-27', '2016-04-28',
                  '2016-04-29', '2016-04-30', '2016-05-01', '2016-05-02',
                  '2016-05-03', '2016-05-04', '2016-05-05', '2016-05-06',
                  '2016-05-07', '2016-05-08', '2016-05-09', '2016-05-10',
                  '2016-05-11', '2016-05-12', '2016-05-13', '2016-05-14',
                  '2016-05-15', '2016-05-16', '2016-05-17', '2016-05-18',
                  '2016-05-19', '2016-05-20', '2016-05-21', '2016-05-22']
    agg_init_date = '2015-10-01'
    X_test_copy = X_test.copy()
    org_feature = feature_config['features']['original'] + \
        ['id', 'all_id', 'date', 'demand']
    tmp_df = X_all[agg_init_date < X_all['date']].copy()
    for test_date in tqdm(test_dates):
        pred_tmp_df = tmp_df[tmp_df['date'] <= test_date][org_feature].copy()
        tmp_feat = features.generate_features(pred_tmp_df)
        pred_tmp_df = pd.concat([pred_tmp_df, tmp_feat], axis=1)
        if cv:
            preds = runner.run_predict_cv(
                pred_tmp_df[pred_tmp_df['date'] == test_date][all_features])
        else:
            preds = runner.run_predict_all(
                pred_tmp_df[pred_tmp_df['date'] == test_date][all_features])
    #     tmp_df[tmp_df['date']==test_date]['demand'] = preds
        row_indexer = X_test_copy[X_test_copy['date'] == test_date].index
        X_test_copy.loc[row_indexer, 'demand'] = preds
    return X_test_copy['demand']
