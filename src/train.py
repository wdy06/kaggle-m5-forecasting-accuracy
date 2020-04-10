import argparse
import os
import random
import shutil

import numpy as np
import pandas as pd
import gc
# import sandesh
from sklearn.model_selection import GroupKFold

import features
import metrics
import mylogger
import preprocessing
import utils
from runner import Runner


parser = argparse.ArgumentParser(description='kaggle data science bowl 2019')
parser.add_argument("--config", "-c", type=str,
                    required=True, help="config path")
parser.add_argument("--feature", "-f", type=str,
                    required=True, help="feature list path")
parser.add_argument("--debug", help="run debug mode",
                    action="store_true")
args = parser.parse_args()

utils.seed_everything()

print(f'on kaggle: {utils.ON_KAGGLE}')
print(utils.DATA_DIR)
print(utils.RESULTS_BASE_DIR)
TARGET_COL = 'demand'

try:
    exp_name = utils.make_experiment_name(args.debug)
    result_dir = utils.RESULTS_BASE_DIR / exp_name
    os.mkdir(result_dir)

    logger = mylogger.get_mylogger(filename=result_dir / 'log')
    # sandesh.send(f'start: {exp_name}')
    logger.debug(f'created: {result_dir}')
    logger.debug('loading data ...')

    train_feat_path = utils.FEATURE_DIR / 'baseline_features.pkl'
    X = utils.load_pickle(train_feat_path)
    print(X.columns)
    # X_test = utils.load_pickle(test_feat_path)

    # new_train = features.add_agg_feature_train(new_train)
    # X_test = features.add_agg_feature_test(X_test, X_test_all)

    # sandesh.send(args.feature)
    features_list = utils.load_yaml(args.feature)
    utils.dump_yaml(features_list, result_dir / 'features_list.yml')
    all_features = features_list['features']
    categorical_feat = features_list['categorical_features']
    # if args.debug:
    #     all_features = [
    #         feat for feat in all_features if feat in new_train.columns]

    logger.debug(all_features)
    logger.debug(f'features num: {len(all_features)}')
    utils.dump_yaml(features_list, result_dir / 'features_list.yml')

    # X_test = X_test[all_features]

    # sandesh.send(args.config)
    config = utils.load_yaml(args.config)
    logger.debug(config)
    utils.dump_yaml(config, result_dir / 'model_config.yml')
    model_params = config['model_params']
    model_params['categorical_feature'] = categorical_feat

    if args.debug:
        model_params['learning_rate'] = 1

    # preprocess for neural network
    # if config['model_class'] == 'ModelNNRegressor':
    #     X, X_test, y, encoder_dict = preprocess.preprocess_for_nn(
    #         X, X_test, y, all_features, categorical_feat)
    #     utils.dump_pickle(encoder_dict, result_dir / 'encoder_dict.pkl')

    # if config['model_class'] == 'ModelLGBMRegressor':
    #     model_params['device'] = 'gpu'
    #     model_params['gpu_platform_id'] = 0
    #     model_params['gpu_device_id'] = 0
    #     model_params['gpu_use_dp'] = True
    # if config['model_class'] == 'ModelXGBRegressor':
    #     model_params['tree_method'] = 'gpu_hist'

    oof = np.zeros(len(X))
    # create folds
    # NFOLDS = 5
    # group_kfold = GroupKFold(n_splits=5)
    fold_indices = []
    all_val_idx = []
    X.reset_index(inplace=True)
    train_idx = X.query('date <= "2016-03-27"').index.tolist()
    val_idx = X.query(
        '"2016-03-27" < date <= "2016-04-24"').index.tolist()
    all_val_idx = all_val_idx + val_idx
    fold_indices.append((train_idx, val_idx))
    # X_train, y_train = X[all_features], X['demand']
    X_train = X[(X['date'] <= '2016-04-24')]
    X_test = X[(X['date'] > '2016-04-24')]
    # X_test = X_test[all_features]
    del X
    gc.collect()
    # all_val_idx = []
    # for i_fold, (train_idx, val_idx) in enumerate(group_kfold.split(new_train, groups=new_train['ins_id'])):
    #     fold_indices.append((train_idx, val_idx))
    #     print(len(train_idx), len(val_idx))
    # utils.reduce_mem_usage(X_train)
    # utils.reduce_mem_usage(X_test)
    utils.dump_pickle(X_train[all_features], result_dir / 'train_x.pkl')
    utils.dump_pickle(X_test[all_features], result_dir / 'test_x.pkl')
    utils.dump_pickle(X_train[TARGET_COL], result_dir / 'train_y.pkl')

    utils.dump_pickle(fold_indices, result_dir / 'fold_indices.pkl')
    # print(X_train.dtypes)
    runner = Runner(run_name='train_cv',
                    x=X_train[all_features],
                    y=X_train[TARGET_COL],
                    model_cls=config['model_class'],
                    params=model_params,
                    metrics=metrics.rmse,
                    save_dir=result_dir,
                    fold_indices=fold_indices
                    )
    val_score, oof_preds = runner.run_train_cv()
    val_score = metrics.rmse(
        oof_preds[all_val_idx], X_train[TARGET_COL][all_val_idx])
    runner.save_importance_cv()

    logger.debug('-' * 30)
    logger.debug(f'OOF RMSE: {val_score}')
    # logger.debug(f'OOF QWK: {val_score}')
    logger.debug('-' * 30)

    # sandesh.send(f'OOF RMSE: {val_rmse}')
    # sandesh.send(f'OOF QWK: {val_score}')

    # process test set
    # X_test = utils.load_pickle(test_feat_path)
    preds = runner.run_predict_cv(X_test[all_features])
    X_test[TARGET_COL] = preds
    # utils.dump_pickle(X_test, result_dir / 'test_x_all.pkl')

    # make final prediction csv
    save_path = result_dir / f'submission_val{val_score:.5f}.csv'
    submission = pd.read_csv(utils.DATA_DIR / 'sample_submission.csv')
    predictions = X_test[['id', 'date', TARGET_COL]]
    predictions = pd.pivot(predictions, index='id',
                           columns='date', values='demand').reset_index()
    predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]
    evaluation_rows = [row for row in submission['id'] if 'evaluation' in row]
    # TODO: predict of evaluation are all 0 now
    evaluation = submission[submission['id'].isin(evaluation_rows)]
    validation = submission[['id']].merge(predictions, on='id')
    final = pd.concat([validation, evaluation])
    final.to_csv(save_path, index=False)
    # submission['accuracy_group'] = np.round(preds).astype('int')
    # submission.to_csv(save_path, index=False)
    logger.debug(f'save to {save_path}')
    # sandesh.send(f'finish: {save_path}')
    # sandesh.send('-' * 30)


except Exception as e:
    print(e)
    # sandesh.send(e)
    logger.exception(e)
    raise e
