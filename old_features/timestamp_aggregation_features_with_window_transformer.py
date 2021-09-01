from sklearn.base import TransformerMixin
import numpy as np
from copy import deepcopy


class TimeStampAggregationWithWindow(TransformerMixin):
    def __init__(self):
        self.window_size = 15
        self.features = ['Q1_window{}'.format(self.window_size), 'Q2_window{}'.format(self.window_size),
                         'Q3_window{}'.format(self.window_size)]

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.features]

    def get_feature_names(self):
        return self.features

    def append_features(self, X):

        timestamp_diff = list(X['timestamp_diff'])
        new_features = list()
        timestamp_features = {
            'Q1': 0,
            'Q2': 0,
            'Q3': 0,
            'Q4': 0
        }
        for index, row in X.iterrows():
            if index == 0 or row['timestamp_diff'] == 0:
                new_features.append(deepcopy(timestamp_features))
            else:
                if index > 30:
                    timestamp_diff_without_zeros = [x for x in timestamp_diff[index - self.window_size:index] if x > 0]
                else:
                    timestamp_diff_without_zeros = [x for x in timestamp_diff[:index] if x > 0]

                timestamp_features['Q1'] = np.quantile(timestamp_diff_without_zeros, .25)
                timestamp_features['Q2'] = np.quantile(timestamp_diff_without_zeros, .50)
                timestamp_features['Q3'] = np.quantile(timestamp_diff_without_zeros, .75)

                new_features.append(deepcopy(timestamp_features))

        for i in range(1, 4):
            q = list()
            for j in range(len(new_features)):
                q.append(new_features[j]['Q{}'.format(i)])
            X['Q{}_window{}'.format(i, self.window_size)] = q

        return X
