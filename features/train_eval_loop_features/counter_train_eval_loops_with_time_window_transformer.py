import ast
import json

from sklearn.base import TransformerMixin
import numpy as np
from copy import deepcopy

def check_if_time_in_window_size(current_time, time_to_check, window_size_in_seconds):
    if current_time - time_to_check < 0:
        return False
    elif current_time - window_size_in_seconds > time_to_check:
        return False
    else:
        return True


class CounterTrainEvalLoopsWithTimeWindow(TransformerMixin):
    def __init__(self):
        self.features = ['number_of_train_eval_loops_in_time_window']

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.features]

    def get_feature_names(self):
        return self.features

    def append_features(self, X):
        configuration_data = json.loads(open("configuration.json", "r").read())
        window_size = configuration_data['time_window_sizes']['train_eval_loop_features']
        number_of_train_eval_loops_in_time_window = list()
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
            X_temp = X_temp[['number_of_train_eval_loops']]
            X_temp = X_temp.drop_duplicates()
            number_of_train_eval_loops_in_time_window.append(len(X_temp) - 1)

        X['number_of_train_eval_loops_in_time_window'] = number_of_train_eval_loops_in_time_window
