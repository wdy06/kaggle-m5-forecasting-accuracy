import argparse
import gc
import os
import random
import shutil

import numpy as np
import pandas as pd
import sandesh
from sklearn.model_selection import GroupKFold

import features
import metrics
import mylogger
import preprocessing
import utils
from create_folds import create_folds
from evaluater import WRMSSEEvaluator
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
    sandesh.send(f'start: {exp_name}')
    logger.debug(f'created: {result_dir}')
    logger.debug('loading data ...')

    train_feat_path = utils.FEATURE_DIR / 'baseline_features.pkl'
    X = utils.load_pickle(train_feat_path)
    print(X.columns)

    features_list = utils.load_yaml(args.feature)
    utils.dump_yaml(features_list, result_dir / 'features_list.yml')
    all_features = features_list['features']
    categorical_feat = features_list['categorical_features']

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
        model_params['learning_rate'] = 10

    # preprocess for neural network
    # if config['model_class'] == 'ModelNNRegressor':
    #     X, X_test, y, encoder_dict = preprocess.preprocess_for_nn(
    #         X, X_test, y, all_features, categorical_feat)
    #     utils.dump_pickle(encoder_dict, result_dir / 'encoder_dict.pkl')

    oof = np.zeros(len(X))
    # create folds
    # fold_indices = create_folds(X)
    fold_indices = utils.load_pickle(utils.FEATURE_DIR / 'fold_indices.pkl')

    if args.debug:
        fold_indices = fold_indices[:2]
    X_train = X[(X['date'] <= '2016-04-24')]
    X_test = X[(X['date'] > '2016-04-24')]

    del X
    gc.collect()
    # utils.reduce_mem_usage(X_train)
    # utils.reduce_mem_usage(X_test)
    utils.dump_pickle(X_train[all_features], result_dir / 'train_x.pkl')
    utils.dump_pickle(X_test[all_features], result_dir / 'test_x.pkl')
    utils.dump_pickle(X_train[TARGET_COL], result_dir / 'train_y.pkl')

    utils.dump_pickle(fold_indices, result_dir / 'fold_indices.pkl')
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
    X_train['pred_demand'] = oof_preds
    # evaluate wrmssee score
    encoder_dict = utils.load_pickle(utils.FEATURE_DIR / 'encoder.pkl')
    score_list = []
    for train_idx, val_idx in fold_indices:
        train_df = preprocessing.melt_to_pivot(
            X_train.iloc[train_idx], 'demand', encoder_dict)
        valid_df = preprocessing.melt_to_pivot(
            X_train.iloc[val_idx], 'demand', encoder_dict)
        wrmssee_evaluater = WRMSSEEvaluator(train_df, valid_df)
        preds_df = preprocessing.melt_to_pivot(
            X_train.iloc[val_idx], 'pred_demand', encoder_dict)
        preds_df = preds_df[[
            i for i in preds_df.columns if i.startswith('d_')]]
        score = wrmssee_evaluater.score(preds_df)
        score_list.append(score)
    print(score_list)
    score = sum(score_list) / len(score_list)
    logger.debug(f'average wrmssee score: {score}')

    runner.save_importance_cv()

    logger.debug('-' * 30)
    logger.debug(f'WRMSSEE score: {score}')
    logger.debug(f'OOF WRMSSEE: {score}')
    logger.debug('-' * 30)

    # process test set
    if args.debug:
        preds = runner.run_predict_cv(X_test[all_features])
    else:
        logger.debug('training all data...')
        # preds = runner.run_predict_cv(X_test[all_features])
        runner.run_train_all()
        preds = runner.run_predict_all(X_test[all_features])
    X_test[TARGET_COL] = preds

    # make final prediction csv
    save_path = result_dir / f'submission_val{score:.5f}.csv'
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
    logger.debug(f'save to {save_path}')
    sandesh.send(f'finish: {save_path}')
    sandesh.send('-' * 30)


except Exception as e:
    print(e)
    sandesh.send(str(e))
    logger.exception(e)
    raise e
