from sklearn.base import TransformerMixin
import json
from copy import deepcopy


class AllActionsCounter(TransformerMixin):
    def __init__(self):
        self.configuration_data = json.loads(open("configuration.json", "r").read())
        self.possible_actions = ['feature_sel', 'FE_binning']
        # self.possible_actions = self.configuration_data['possible_actions']
        # self.possible_actions.remove('train')
        # self.possible_actions.remove('eval')
        self.counter_columns = ['all_actions_counter']

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.counter_columns]

    def get_feature_names(self):
        return self.counter_columns

    def append_features(self, X):
        labels = X['label'].values
        counter_all = 1
        all_actions_counter = list()

        for label in labels:
            all_actions_counter.append(counter_all)
            if label in self.possible_actions:
                counter_all += 1

        X['all_actions_counter'] = all_actions_counter

        return X
