import json
from sklearn.base import TransformerMixin
from copy import deepcopy
import ast


def check_if_time_diff_greater_than_window_size(current_time, time_to_check, window_size_in_seconds):
    if current_time - time_to_check > window_size_in_seconds:
        return True, current_time - time_to_check
    else:
        return False, 0


class InactivityMoments(TransformerMixin):
    def __init__(self):
        self.column_names = ['inactivity_flag', 'inactivity_length']

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.column_names]

    def get_feature_names(self):
        return self.column_names

    def append_features(self, X):
        configuration_data = json.loads(open("configuration.json", "r").read())
        window_size = configuration_data['time_window_sizes']['length_period_to_be_considered_inactivity_moment']

        inactivity_moments = [0]
        inactivity_lens = [0]
        for i in range(1, len(X)):
            time_from_start = X['time_from_start_sec'].iloc[i]
            time_to_check = X['time_from_start_sec'].iloc[i - 1]
            inactivity_moment_flag, inactivity_length = check_if_time_diff_greater_than_window_size(time_from_start,
                                                                                                    time_to_check,
                                                                                                    window_size)
            if inactivity_moment_flag:
                inactivity_moments.append(1)
                inactivity_lens.append(inactivity_length)
            else:
                inactivity_moments.append(0)
                inactivity_lens.append(0)

        X['inactivity_flag'] = inactivity_moments
        X['inactivity_len'] = inactivity_lens
        return X
