import os
import random as rn

import keras
import keras.backend as K
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.optimizers import Adam

import utils
from models.model import Model
from models.keras_layer_normalization.layer_normalization import LayerNormalization

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(7)
rn.seed(7)


class ModelNNRegressor(Model):
    def __init__(self, run_fold_name, params):
        super().__init__(run_fold_name, params)

    def fit(self, train_x, train_y, valid_x=None, valid_y=None):

        input_dim = train_x.values.shape[1]
        learning_rate = self.params['learning_rate']
        loss = self.params['loss']
        epochs = self.params['epochs']
        self.model = Sequential([
            layers.Dense(200, activation='relu', input_dim=input_dim),
            # LayerNormalization(),
            layers.Dropout(0.3),
            layers.Dense(100, activation='relu'),
            # LayerNormalization(),
            layers.Dropout(0.3),
            layers.Dense(50, activation='relu'),
            # LayerNormalization(),
            layers.Dropout(0.3),
            layers.Dense(25, activation='relu'),
            # LayerNormalization(),
            layers.Dropout(0.3),
            layers.Dense(1)
        ])

        self.model.compile(optimizer=Adam(
            lr=learning_rate), loss=loss)
        callbacks = []
        # callbacksa.append(ModelCheckpoint(save_best_only=True))
        callbacks.append(EarlyStopping(patience=10))
        if (valid_x is None) or (valid_y is None):
            self.model.fit(train_x, train_y, epochs=epochs,
                           callbacks=callbacks)
        else:
            self.model.fit(train_x, train_y, epochs=epochs,
                           validation_data=(valid_x, valid_y),
                           callbacks=callbacks)

    def predict(self, test_x):
        return self.model.predict(test_x).flatten()

    def save_model(self, path):
        # utils.dump_pickle(self.model, path)
        self.model.save(path)

    def load_model(self, path):
        # self.model = utils.load_pickle(path)
        self.model = keras.models.load_model(path,
                                             custom_objects={'LayerNormalization': LayerNormalization})
