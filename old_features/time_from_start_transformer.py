from sklearn.base import TransformerMixin
import json
import pickle
from sklearn.preprocessing import OneHotEncoder


class TimeFromStart(TransformerMixin):
    def __init__(self):
        self.features = ['time_from_start_sec']

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.features]

    def get_feature_names(self):
        return self.features

    def append_features(self, X):
        start_time = X['timestamp'].iloc[0]
        X['time_from_start_sec'] = X[['timestamp']].sub(start_time)
        X['time_from_start_min'] = X['time_from_start_sec']/60

        return X
