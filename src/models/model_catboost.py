import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

import utils
from models.model import Model


class ModelCatBoostRegressor(Model):

    def fit(self, train_x, train_y, valid_x=None, valid_y=None):
        _params = self.params.copy()

        cat_features = _params.pop('categorical_feature', [])

        self.model = CatBoostRegressor(**_params)
        if (valid_x is None) or (valid_y is None):
            self.model.fit(train_x, train_y,
                           verbose=100, cat_features=cat_features)
        else:
            if _params.get('boosting_type') == 'dart':
                self.model.fit(train_x, train_y, eval_set=(valid_x, valid_y),
                               verbose=100, cat_features=cat_features)
            else:
                self.model.fit(train_x, train_y, eval_set=(valid_x, valid_y),
                               verbose=100, early_stopping_rounds=100,
                               cat_features=cat_features)

    def predict(self, test_x):
        return self.model.predict(test_x)

    def save_model(self, path):
        utils.dump_pickle(self.model, path)

    def load_model(self, path):
        self.model = utils.load_pickle(path)
