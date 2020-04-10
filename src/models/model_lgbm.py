import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

import utils
from models.model import Model


class ModelLGBMClassifier(Model):

    def fit(self, train_x, train_y, valid_x=None, valid_y=None):
        _params = self.params.copy()

        cat_features = _params.pop('categorical_feature', [])

        self.model = LGBMClassifier(**_params)
        if (valid_x is None) or (valid_y is None):
            self.model.fit(train_x, train_y,
                           verbose=100, categorical_feature=cat_features)
        else:
            self.model.fit(train_x, train_y, eval_set=(valid_x, valid_y),
                           verbose=100, early_stopping_rounds=100,
                           categorical_feature=cat_features)

    def predict(self, test_x):
        return self.model.predict(
            test_x, num_iteration=self.model.best_iteration_)

    def save_model(self, path):
        utils.dump_pickle(self.model, path)

    def load_model(self, path):
        self.model = utils.load_pickle(path)


class ModelLGBMRegressor(Model):

    def fit(self, train_x, train_y, valid_x=None, valid_y=None):
        _params = self.params.copy()

        cat_features = _params.pop('categorical_feature', [])

        self.model = LGBMRegressor(**_params)
        if (valid_x is None) or (valid_y is None):
            self.model.fit(train_x, train_y,
                           verbose=100, categorical_feature=cat_features)
        else:
            if _params.get('boosting_type') == 'dart':
                self.model.fit(train_x, train_y, eval_set=(valid_x, valid_y),
                               verbose=100, categorical_feature=cat_features)
            else:
                self.model.fit(train_x, train_y, eval_set=(valid_x, valid_y),
                               verbose=100, early_stopping_rounds=100,
                               categorical_feature=cat_features)

    def predict(self, test_x):
        return self.model.predict(
            test_x, num_iteration=self.model.best_iteration_)

    def get_importance(self):
        if self.columns is None:
            raise ValueError('self.columns must be set.')
        else:
            df = pd.DataFrame()
            df['feature'] = self.columns
            df['gain'] = self.model.feature_importances_
            return df

    def save_model(self, path):
        utils.dump_pickle(self.model, path)

    def load_model(self, path):
        self.model = utils.load_pickle(path)
