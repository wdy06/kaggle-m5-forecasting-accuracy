import argparse
import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd

import features
import metrics
import preprocess
import utils
from dataset import DSB2019Dataset
from optimizedrounder import OptimizedRounder, HistBaseRounder
from runner import Runner

parser = argparse.ArgumentParser(description='kaggle data science bowl 2019')
parser.add_argument("--debug", help="run debug mode",
                    action="store_true")
parser.add_argument("--input-dir", type=str, help="input directory path")
args = parser.parse_args()

print(f'on kaggle: {utils.ON_KAGGLE}')
print(utils.DATA_DIR)
print(utils.RESULTS_BASE_DIR)


input_dir = Path(args.input_dir)
print(f'input dir: {input_dir}')
print('loading data ...')

train_feat_path = utils.FEATURE_DIR / 'train_features.pkl'
test_feat_path = utils.FEATURE_DIR / 'test_features.pkl'
all_test_feat_path = utils.FEATURE_DIR / 'all_test_features.pkl'
if args.debug:
    train_feat_path = utils.FEATURE_DIR / 'train_features_debug.pkl'
    test_feat_path = utils.FEATURE_DIR / 'test_features_debug.pkl'
    all_test_feat_path = utils.FEATURE_DIR / 'all_test_features_debug.pkl'


train = DSB2019Dataset(mode='train')
event_code_list = list(train.main_df.event_code.unique())
event_id_list = list(train.main_df.event_id.unique())

del train
gc.collect()

features_list = utils.load_yaml(input_dir / 'features_list.yml')
all_features = features_list['features']
cat_features = features_list['categorical_features']
print(all_features)
print(f'features num: {len(all_features)}')
# X, y = new_train[all_features], new_train['accuracy_group']

config_path = input_dir / 'model_config.yml'
config = utils.load_yaml(config_path)
print(config)
model_params = config['model_params']
model_params['categorical_feature'] = cat_features

fold_indices = utils.load_pickle(input_dir / 'fold_indices.pkl')

runner = Runner(run_name='train_cv',
                x=None,
                y=None,
                model_cls=config['model_class'],
                params=model_params,
                metrics=metrics.qwk,
                save_dir=input_dir,
                fold_indices=fold_indices
                )

activities_map = utils.load_json(utils.CONFIG_DIR / 'activities_map.json')
# feature_mapper = utils.load_json(input_dir / 'feature_mapper.json')
win_code = utils.make_win_code(activities_map)

# process test set
if utils.ON_KAGGLE:
    test = DSB2019Dataset(mode='test')
    test = preprocess.preprocess_dataset(test)
    X_test, all_test_history = features.generate_features_by_acc(
        test.main_df, win_code, event_code_list, event_id_list, mode='test')
    # for feat_name in feature_mapper.keys():
    #     X_test[feat_name] = X_test['session_title'].map(
    #         feature_mapper[feat_name])
    del test
    gc.collect()
else:
    X_test = utils.load_pickle(test_feat_path)
    all_test_history = utils.load_pickle(all_test_feat_path)

X_test = features.add_feature(X_test, activities_map)
X_test = features.add_agg_feature_test(X_test, all_test_history)
# adjust data
if os.path.exists(input_dir / 'adjust.json'):
    print('adjust !!!')
    adjust_dict = utils.load_json(input_dir / 'adjust.json')
    for key, factor in adjust_dict.items():
        # print(f'{key}: {factor}')
        X_test[key] *= factor

X_test = X_test[all_features]

if config['model_class'] == 'ModelNNRegressor':
    print('preprocessing for nn ...')
    encoder_dict = utils.load_pickle(input_dir / 'encoder_dict.pkl')
    X_test = preprocess.preprocess_for_nn_from_encoder_dict(
        X_test, all_features, cat_features, encoder_dict)
# print(X_test)
# preds = runner.run_predict_all(X_test)
preds = runner.run_predict_cv(X_test)
if config['model_class'] == 'ModelNNRegressor':
    print('post processing for nn ...')
    preds = preprocess.postprocess_for_nn(preds, encoder_dict)
print(preds)
if config['task'] == 'regression':
    # optR = OptimizedRounder()
    optR = HistBaseRounder()
    best_coef = utils.load_pickle(input_dir / 'best_coef.pkl')
    preds = optR.predict(preds, best_coef)
# save_path = result_dir / f'submission_val{val_score:.5f}.csv'
if utils.ON_KAGGLE:
    save_path = 'submission.csv'
    submission = pd.read_csv(utils.DATA_DIR / 'sample_submission.csv')
    submission['accuracy_group'] = np.round(preds).astype('int')
    submission.to_csv(save_path, index=False)
print('finish !!!')
