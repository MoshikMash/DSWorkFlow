from sklearn.base import TransformerMixin
import json
import pickle
from sklearn.preprocessing import OneHotEncoder


class Metric(TransformerMixin):
    def __init__(self):
        self.features = ['metric']

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.features]

    def get_feature_names(self):
        return self.features

    def append_features(self, X):
        score = list(X['score'])
        score.insert(0, -1)
        score.pop()
        new_score = list()
        for x in score:
            if x == -1:
                new_score.append('0')
            else:
                new_score.append(x)
        X['score'] = score
        return X
