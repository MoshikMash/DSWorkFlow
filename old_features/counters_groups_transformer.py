import pandas as pd
from sklearn.base import TransformerMixin
import json
from copy import deepcopy


class CountersGroups(TransformerMixin):
    def __init__(self):
        self.configuration_data = json.loads(open("configuration.json", "r").read())
        self.counter_columns = list()
        self.counter_groups_columns = list()
        actions = self.configuration_data['possible_actions']
        for action in actions:
            self.counter_columns.append('{}_counter'.format(action))
            self.counter_groups_columns.append('{}_groups_counter'.format(action))

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.counter_groups_columns]

    def get_feature_names(self):
        return self.counter_groups_columns

    def append_features(self, X):
        labels = X['label'].values
        possible_actions = self.configuration_data['possible_actions']
        counters_dictionary = {key: 0 for key in possible_actions}

        sum1 = 0
        # the first element's value should be empty dictionary
        triggers = [deepcopy(counters_dictionary)]

        for label in labels:
            if label in counters_dictionary:
                counters_dictionary[label] += 1
                sum1 += 1
            triggers.append(deepcopy(counters_dictionary))

        # drop the last dictionary
        triggers.pop()

        X['trigger_counter'] = triggers

        # create column for each trigger
        for action in counters_dictionary:
            l = []
            for index, row in X.iterrows():
                l.append(row['trigger_counter'][action])
            X['{}_counter'.format(action)] = l

        actions_bins = self.configuration_data['counters_groups_bins']
        for action in possible_actions:
            bins = actions_bins[action]
            X['{}_groups_counter'.format(action)] = pd.cut(x=X['{}_counter'.format(action)], bins=bins)

        return X
