import json

from sklearn.base import TransformerMixin
from copy import deepcopy
import ast


def check_if_time_in_window_size(current_time, time_to_check, window_size_in_seconds):
    if current_time - time_to_check < 0:
        return False
    elif current_time - window_size_in_seconds > time_to_check:
        return False
    else:
        return True


class PerformanceImprovementWithTimeWindowSize(TransformerMixin):
    def __init__(self):
        self.metrics = ['accuracy', 'auc']
        self.column_names = list()
        for metric in self.metrics:
            self.column_names.append('score_improved_in_time_window_{}'.format(metric))

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.column_names]

    def get_feature_names(self):
        return self.column_names

    @staticmethod
    def append_score_was_improved_column(X, metric):
        configuration_data = json.loads(open("configuration.json", "r").read())
        window_size = configuration_data['time_window_sizes']['performance_features']

        improved_score_indicators_in_window = list()
        for index, row in X.iterrows():
            X_temp = deepcopy(X)
            time_from_start = row['time_from_start_sec']
            include_in_the_window = list()
            for index2, row2 in X.iterrows():
                time_to_check = row2['time_from_start_sec']
                if check_if_time_in_window_size(time_from_start, time_to_check, window_size):
                    include_in_the_window.append(1)
                else:
                    include_in_the_window.append(0)
            X_temp['in_window'] = include_in_the_window
            X_temp = X_temp[X_temp['in_window'] == 1]
            if 1 in X_temp['score_improved_{}'.format(metric)].values:
                improved_score_indicators_in_window.append(1)
            else:
                improved_score_indicators_in_window.append(0)

        X['score_improved_in_time_window_{}'.format(metric)] = improved_score_indicators_in_window
        return X

    def append_features(self, X):
        for metric in self.metrics:
            X = self.append_score_was_improved_column(X, metric)
