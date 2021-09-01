import ast

from sklearn.base import TransformerMixin
import json
from copy import deepcopy


def check_if_time_in_window_size(current_time, time_to_check, window_size_in_seconds):
    if current_time - time_to_check < 0:
        return False
    elif current_time - window_size_in_seconds > time_to_check:
        return False
    else:
        return True


class CountersWithTimeWindowSizeActionChangedBetweenTrainEvalLoops(TransformerMixin):
    def __init__(self):
        self.configuration_data = json.loads(open("configuration.json", "r").read())
        self.possible_actions = self.configuration_data['possible_actions']
        self.counter_columns = list()
        actions = self.possible_actions
        for action in actions:
            self.counter_columns.append(
                '{}_counter_with_time_window_size_action_was_changed_between_train_eval_loops'.format(action))

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.counter_columns]

    def get_feature_names(self):
        return self.counter_columns

    def append_features(self, X):
        configuration_data = json.loads(open("configuration.json", "r").read())
        window_size = configuration_data['time_window_sizes']['performance_features']

        for action in self.possible_actions:
            action_was_changed_indicators_in_window = list()
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
                X_temp = X_temp[['in_window', '{}_action_was_changed_between_train_eval_loops'.format(action),
                                 'number_of_train_eval_loops']]
                X_temp = X_temp.drop_duplicates()
                X_temp = X_temp[X_temp['{}_action_was_changed_between_train_eval_loops'.format(action)]]
                action_was_changed_indicators_in_window.append(len(X_temp))

            X['{}_counter_with_time_window_size_action_was_changed_between_train_eval_loops'.format(
                action)] = action_was_changed_indicators_in_window
