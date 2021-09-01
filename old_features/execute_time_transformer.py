import pandas as pd
from sklearn.base import TransformerMixin
import json
from copy import deepcopy
from sklearn.preprocessing import OneHotEncoder


class ExecuteTime(TransformerMixin):
    def __init__(self):
        self.configuration_data = json.loads(open("configuration.json", "r").read())
        self.features = None
        self.enc = False

    def fit(self, X):
        self.enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.enc.fit(X[['execute_time_groups']])
        self.features = self.enc.get_feature_names(list(set(X[['execute_time_groups']])))
        return self

    def transform(self, X):
        return self.enc.transform(X[['execute_time_groups']])

    def get_feature_names(self):
        return self.features

    def append_features(self, X):
        bins = self.configuration_data['execute_time_bins']
        X['execute_time_groups'] = pd.cut(x=X['execTime'], bins=bins)
        return X
