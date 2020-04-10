import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from models.helper import MODEL_MAP
import utils


class Runner:
    def __init__(self, run_name, x, y,
                 model_cls, params, metrics, save_dir, fold_indices=None):
        super().__init__()
        self.run_name = run_name
        self.x = x
        self.y = y
        self.model_cls = model_cls
        self.params = params
        self.metrics = metrics
        self.save_dir = save_dir
        self.fold_indeices = fold_indices

    def train(self, train_x, train_y, val_x=None, val_y=None):
        validation = True
        if (val_x is None) or (val_y is None):
            validation = False
        model = self.build_model()
        if validation:
            model.fit(train_x, train_y, val_x, val_y)
            y_pred = model.predict(val_x)
            return model, y_pred
        else:
            model.fit(train_x, train_y)
            return model, None

    def run_train_cv(self):
        oof_preds = np.zeros(len(self.y))
        for i_fold, (trn_idx, val_idx) in enumerate(self.fold_indeices):
            train_x = self.x.iloc[trn_idx]
            train_y = self.y.iloc[trn_idx]
            val_x = self.x.iloc[val_idx]
            val_y = self.y.iloc[val_idx]
            model, y_pred = self.train(train_x, train_y, val_x, val_y)
            model.save_model(
                self.save_dir / f'{self.run_name}_fold{i_fold}.pkl')
            oof_preds[val_idx] = y_pred
        oof_score = self.metrics(self.y, oof_preds)
        return oof_score, oof_preds

    def run_predict_cv(self, test_x):
        preds = np.zeros(len(test_x))
        for i_fold, _ in enumerate(self.fold_indeices):
            model = self.build_model()
            model.load_model(
                self.save_dir / f'{self.run_name}_fold{i_fold}.pkl')
            preds += model.predict(test_x)
        preds /= len(self.fold_indeices)
        return preds

    def run_train_all(self):
        model, _ = self.train(self.x, self.y)
        model.save_model(self.save_dir / f'{self.run_name}_all.pkl')

    def run_predict_all(self, test_x):
        model = self.build_model()
        model.load_model(self.save_dir / f'{self.run_name}_all.pkl')
        preds = model.predict(test_x)
        return preds

    def get_oof_preds(self):
        oof_preds = np.zeros(len(self.y))
        for i_fold, (trn_idx, val_idx) in enumerate(self.fold_indeices):
            val_x = self.x.iloc[val_idx]
            model = self.build_model()
            model.load_model(
                self.save_dir / f'{self.run_name}_fold{i_fold}.pkl')
            oof_preds[val_idx] = model.predict(val_x)
        return oof_preds, self.y

    def save_importance_cv(self):
        imp_df = pd.DataFrame()
        for i_fold, _ in enumerate(self.fold_indeices):
            model = self.build_model()
            model.load_model(
                self.save_dir / f'{self.run_name}_fold{i_fold}.pkl')
            model.set_columns(self.x.columns)
            df = model.get_importance()
            # print(df)
            imp_df = pd.concat([imp_df, df], axis=0, sort=False)

        utils.save_importances(imp_df, self.save_dir)

    def build_model(self):
        return MODEL_MAP[self.model_cls]('test_build', self.params)
