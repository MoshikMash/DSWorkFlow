import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
import json
from sklearn.preprocessing import OneHotEncoder


class PerformanceGroups(TransformerMixin):
    def __init__(self):
        self.configuration_data = json.loads(open("configuration.json", "r").read())
        self.metrics = self.configuration_data['performance_metrics']
        self.column_names = list()
        self.column_names_groups = list()
        for metric in self.metrics:
            self.column_names_groups.append('best_score_groups_{}'.format(metric))

        # for metric in self.metrics:
        #     self.column_names.append('best_score_{}'.format(metric))

        self.enc = None

    def fit(self, X):
        self.enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.enc.fit(X[self.column_names_groups])
        return self

    def transform(self, X):
        return self.enc.transform(X[self.column_names_groups])

    def get_feature_names(self):
        return self.enc.get_feature_names(self.column_names_groups)

    def append_features(self, X):
        for metric in self.metrics:
            if 'best_score_{}'.format(metric) in list(X):
                X['best_score_{}'.format(metric)] = np.where(X['previous_action_0'] == 'eval',
                                                             X['best_score_{}'.format(metric)], 0)

                best_score_bins = self.configuration_data['best_score_bins'][metric]
                X['best_score_groups_{}'.format(metric)] = pd.cut(x=X['best_score_{}'.format(metric)],
                                                                  bins=best_score_bins)
        return X
