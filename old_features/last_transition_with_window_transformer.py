from itertools import product

from sklearn.base import TransformerMixin
import json
from copy import deepcopy
from sklearn.preprocessing import OneHotEncoder


class LastTransitionWithWindow(TransformerMixin):
    def __init__(self):
        self.configuration_data = json.loads(open("configuration.json", "r").read())
        self.features = None
        self.enc = False

    def fit(self, X):
        self.enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.enc.fit(X[['last_transition_with_window']])
        self.features = self.enc.get_feature_names(list(set(X[['last_transition_with_window']])))
        return self

    def transform(self, X):
        return self.enc.transform(X[['last_transition_with_window']])

    def get_feature_names(self):
        return self.features

    def append_features(self, X):
        window_size = self.configuration_data['previous_actions_window_size']
        labels = X['label'].values
        possible_actions = self.configuration_data['possible_actions']
        possible_actions.append('-')
        transitions_dictionary = {ele: '-' for ele in product(possible_actions, repeat=window_size)}
        triggers = list()

        # it doesn't matter because all the vales are '-'
        previous_labels = ['-'] * window_size
        for i, label in enumerate(labels):
            triggers.append(transitions_dictionary[tuple(previous_labels)])
            transitions_dictionary[tuple(previous_labels)] = label
            previous_labels.insert(0, label)
            previous_labels.pop()

        X['last_transition_with_window'] = triggers

        return X
