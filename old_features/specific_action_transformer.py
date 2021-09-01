from sklearn.base import TransformerMixin
import json
from copy import deepcopy
from sklearn.preprocessing import OneHotEncoder


class SpecificAction(TransformerMixin):
    def __init__(self):
        self.configuration_data = json.loads(open("configuration.json", "r").read())
        self.enc = OneHotEncoder()

    def fit(self, X):
        self.enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.enc.fit(X[['specific_action']])
        return self

    def transform(self, X):
        return self.enc.transform(X[['specific_action']])

    def get_feature_names(self):
        return self.enc.get_feature_names()

    def append_features(self, X):
        return X

