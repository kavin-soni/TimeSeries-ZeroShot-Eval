import xgboost as xgb
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import tensorflow as tf

class XGBoostBaseline:
    def __init__(self, params=None):
        self.params = params if params else {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'eta': 0.05,
            'max_depth': 8,
            'seed': 42,
            'tree_method': 'hist'
        }
        self.model = None

    def train(self, X_train, y_train, X_val, y_val):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        evallist = [(dtrain, 'train'), (dval, 'val')]
        self.model = xgb.train(self.params, dtrain, num_boost_round=1000, 
                               evals=evallist, early_stopping_rounds=50, verbose_eval=False)

    def predict(self, X_test):
        dtest = xgb.DMatrix(X_test)
        return self.model.predict(dtest, iteration_range=(0, self.model.best_iteration))

class LSTMBaseline:
    def __init__(self, lookback, horizon, units=64):
        self.lookback = lookback
        self.horizon = horizon
        self.model = Sequential([
            Input(shape=(lookback, 1)),
            LSTM(units=units, activation='tanh'),
            Dense(horizon)
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def train(self, X_train, y_train, epochs=5, batch_size=64):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self, X_test):
        return self.model.predict(X_test, verbose=0)

def run_snaive(y_train, horizon, seasonality):
    """
    Seasonal Naive Baseline.
    """
    if len(y_train) < seasonality:
        return np.tile(y_train[-1:], horizon)
    else:
        last_season = y_train[-seasonality:]
        repetitions = int(np.ceil(horizon / seasonality))
        return np.tile(last_season, repetitions)[:horizon]
