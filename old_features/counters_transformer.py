from sklearn.base import TransformerMixin
import json
from copy import deepcopy


class Counters(TransformerMixin):
    def __init__(self):
        self.configuration_data = json.loads(open("configuration.json", "r").read())
        self.possible_actions = self.configuration_data['possible_actions']
        self.counter_columns = list()
        actions = self.configuration_data['possible_actions']
        for action in actions:
            self.counter_columns.append('{}_counter'.format(action))

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.counter_columns]

    def get_feature_names(self):
        return self.counter_columns

    def append_features(self, X):
        labels = X['label'].values
        counters_dictionary = {key: 0 for key in self.possible_actions}

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

        return X
