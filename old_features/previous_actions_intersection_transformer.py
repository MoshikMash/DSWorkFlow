from sklearn.base import TransformerMixin
import json
import pickle
import numpy as np
from actionPrediction.features.previous_actions_transformer import PreviousActions
from copy import deepcopy
from sklearn.preprocessing import OneHotEncoder


class PreviousActionsIntersection(TransformerMixin):
    def __init__(self):
        self.features = None
        self.enc = False

    def fit(self, X):
        self.enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.enc.fit(X[['previous_actions_intersection']])
        self.features = self.enc.get_feature_names(list(set(X[['previous_actions_intersection']])))
        return self

    def transform(self, X):
        return self.enc.transform(X[['previous_actions_intersection']])

    def get_feature_names(self):
        return self.features

    def append_features(self, X):
        previous_actions_transformer_obj = PreviousActions()
        previous_actions_transformer_obj.append_features(X)
        previous_actions_features = previous_actions_transformer_obj.get_column_names_before_one_hot_encoding()
        X['previous_actions_intersection'] = X[previous_actions_features[0]]
        for i in range(1, len(previous_actions_features)):
            X['previous_actions_intersection'] = ['{}_{}'.format(a, b) for a, b in
                                                  zip(X['previous_actions_intersection'],
                                                      X[previous_actions_features[i]])]

        return X['previous_actions_intersection']
