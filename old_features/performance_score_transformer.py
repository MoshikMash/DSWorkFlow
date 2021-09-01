import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
import json
from sklearn.preprocessing import OneHotEncoder


class Performance(TransformerMixin):
    def __init__(self):
        self.configuration_data = json.loads(open("configuration.json", "r").read())
        self.metrics = self.configuration_data['performance_metrics']
        self.column_names = list()

        for metric in self.metrics:
            self.column_names.append('best_score_{}'.format(metric))

        self.enc = None

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.get_feature_names()]

    def get_feature_names(self):
        return self.column_names

    def append_features(self, X):
        return X
