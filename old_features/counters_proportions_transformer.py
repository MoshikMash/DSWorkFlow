from sklearn.base import TransformerMixin
import json
from copy import deepcopy


class CountersProportions(TransformerMixin):
    def __init__(self):
        self.configuration_data = json.loads(open("configuration.json", "r").read())
        self.possible_actions = self.configuration_data['possible_actions']
        self.possible_actions.remove('train')
        self.possible_actions.remove('eval')
        self.counter_columns = list()
        actions = self.possible_actions
        for action in actions:
            self.counter_columns.append('{}_proportions_counter'.format(action))

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.counter_columns]

    def get_feature_names(self):
        return self.counter_columns

    def append_features(self, X):
        labels = X['label'].values
        counters_dictionary = {key: 0 for key in self.possible_actions}
        counters_proportions_dictionary = {key: 0 for key in self.possible_actions}
        counter_all = 0

        sum1 = 0
        # the first element's value should be empty dictionary
        triggers = [deepcopy(counters_dictionary)]

        for label in labels:
            if label in counters_dictionary:
                counter_all += 1
                counters_dictionary[label] += 1
                counters_proportions_dictionary[label] = counters_dictionary[label] / counter_all
                sum1 += 1
            triggers.append(deepcopy(counters_proportions_dictionary))

        # drop the last dictionary
        triggers.pop()

        X['trigger_proportions_counter'] = triggers

        # create column for each trigger
        for action in counters_dictionary:
            l = []
            for index, row in X.iterrows():
                l.append(row['trigger_proportions_counter'][action])
            X['{}_proportions_counter'.format(action)] = l

        return X
