from sklearn.base import TransformerMixin
import json
import pickle
from sklearn.preprocessing import OneHotEncoder


class MLalgo(TransformerMixin):
    def __init__(self):
        self.features = ['ml_algo']

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.features]

    def get_feature_names(self):
        return self.features

    def append_features(self, X):
        ml_algo = list(X['ml_algo'])
        ml_algo.insert(0, -1)
        ml_algo.pop()
        X['ml_algo'] = ml_algo
        return X
