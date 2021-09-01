import json

from sklearn.base import TransformerMixin
from copy import deepcopy


class CounterAtLeastOneImprovedScore(TransformerMixin):
    def __init__(self):
        self.metrics = ['accuracy', 'auc']
        self.column_names = ['at_lease_one_improved_score_in_time_window']

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.column_names]

    def get_feature_names(self):
        return self.column_names

    def append_features(self, X):
        for i, metric in enumerate(self.metrics):
            if i == 0:
                X['at_lease_one_improved_score_in_time_window'] = deepcopy(
                    X['score_improved_in_time_window_{}'.format(metric)])
            else:
                X['at_lease_one_improved_score_in_time_window'] += X['score_improved_in_time_window_{}'.format(metric)]
