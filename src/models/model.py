import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod


class Model(metaclass=ABCMeta):
    def __init__(self, run_fold_name: str, params: dict):
        """コンストラクタ
        :param run_fold_name: ランの名前とfoldの番号を組み合わせた名前
        :param params: ハイパーパラメータ
        """
        self.run_fold_name = run_fold_name
        self.params = params
        self.model = None
        self.columns = None

    def set_columns(self, columns):
        self.columns = columns

    @abstractmethod
    def fit(self, train_x, train_y, valid_x=None, valid_y=None):
        """モデルの学習を行い、学習済のモデルを保存する
        :param tr_x: 学習データの特徴量
        :param tr_y: 学習データの目的変数
        :param va_x: バリデーションデータの特徴量
        :param va_y: バリデーションデータの目的変数
        """
        pass

    @abstractmethod
    def predict(self, test_x):
        """学習済のモデルでの予測値を返す
        :param te_x: バリデーションデータやテストデータの特徴量
        :return: 予測値
        """
        pass

    @abstractmethod
    def save_model(self, path):
        """モデルの保存を行う"""
        pass

    @abstractmethod
    def load_model(self, path):
        """モデルの読み込みを行う"""
        pass
