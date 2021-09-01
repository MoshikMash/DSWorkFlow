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


class TrainRepetitionCounterWithWindow(TransformerMixin):
    def __init__(self):
        self.column_names = ['train_repetition_counter']

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.column_names]

    def get_feature_names(self):
        return self.column_names

    def append_features(self, X):
        configuration_data = json.loads(open("configuration.json", "r").read())
        window_size = configuration_data['time_window_sizes']['irrational_behavior_features']

        train_repetitions_counter = []
        for index, row in X.iterrows():
            X_temp = deepcopy(X)
            time_from_start = row['time_from_start_sec']
            include_in_the_window = []
            for index2, row2 in X.iterrows():
                time_to_check = row2['time_from_start_sec']
                if check_if_time_in_window_size(time_from_start, time_to_check, window_size):
                    include_in_the_window.append(1)
                else:
                    include_in_the_window.append(0)
            X_temp['in_window'] = include_in_the_window
            X_temp = X_temp[X_temp['in_window'] == 1]
            X_temp = X_temp[X_temp['train_repetition_flag'] == 1]
            train_repetitions_counter.append(len(X_temp))

        X['train_repetition_counter'] = train_repetitions_counter
