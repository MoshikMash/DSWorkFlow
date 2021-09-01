from sklearn.base import TransformerMixin
import json
from copy import deepcopy
from sklearn.preprocessing import OneHotEncoder


class LastTransition(TransformerMixin):
    def __init__(self):
        self.configuration_data = json.loads(open("configuration.json", "r").read())
        self.features = None
        self.enc = False

    def fit(self, X):
        self.enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.enc.fit(X[['last_transition']])
        self.features = self.enc.get_feature_names(list(set(X[['last_transition']])))
        return self

    def transform(self, X):
        return self.enc.transform(X[['last_transition']])

    def get_feature_names(self):
        return self.features

    def append_features(self, X):
        labels = X['label'].values
        possible_actions = self.configuration_data['possible_actions']
        transitions_dictionary = {key: '-' for key in possible_actions}
        triggers = list()

        # it doesn't matter because all the vales are '-'
        previous_label = 'train'
        for i, label in enumerate(labels):
            triggers.append(transitions_dictionary[previous_label])
            if i > 0:
                transitions_dictionary[previous_label] = label
            previous_label = label

        X['last_transition'] = triggers

        return X
