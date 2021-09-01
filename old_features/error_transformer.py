from sklearn.base import TransformerMixin
import json
import pickle
from sklearn.preprocessing import OneHotEncoder


class Error(TransformerMixin):
    def __init__(self):
        self.configuration_data = json.loads(open("configuration.json", "r").read())
        self.features = ['error_flag']

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.features]

    def get_feature_names(self):
        return self.features

    def append_features(self, X):
        has_error = list(X['has_error'])
        has_error.insert(0, -1)
        has_error.pop()
        X['error_flag'] = has_error
        return X
