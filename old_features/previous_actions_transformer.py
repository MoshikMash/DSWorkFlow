import pandas as pd
from sklearn.base import TransformerMixin
import json
from copy import deepcopy
from sklearn.preprocessing import OneHotEncoder


class PreviousActions(TransformerMixin):
    def __init__(self):
        self.configuration_data = json.loads(open("configuration.json", "r").read())
        self.features = None
        self.enc = None

    def get_column_names_before_one_hot_encoding(self):
        window_size = self.configuration_data["previous_actions_window_size"]
        previous_actions_columns = list()
        for i in range(window_size):
            previous_actions_columns.append('previous_action_{}'.format(i))

        return previous_actions_columns

    def fit(self, X):
        self.enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
        columns_to_encode = self.get_column_names_before_one_hot_encoding()
        self.enc.fit(X[columns_to_encode])
        self.features = self.enc.get_feature_names(list(set(X[columns_to_encode])))
        return self

    def transform(self, X):
        columns_to_encode = self.get_column_names_before_one_hot_encoding()
        return self.enc.transform(X[columns_to_encode])

    def get_feature_names(self):
        return self.enc.get_feature_names()

    def append_features(self, X):
        actions_history = list()
        window_size = self.configuration_data["previous_actions_window_size"]
        previous_actions = list(['-'] * window_size)
        i = 0
        for index, row in X.iterrows():
            actions_history.append(deepcopy(previous_actions))
            previous_actions.insert(0, str(row['label']))
            previous_actions.pop()

        X['previous_actions'] = actions_history

        for i in range(window_size):
            X['previous_action_{}'.format(i)] = ''

        # create column for each trigger
        for index, row in X.iterrows():
            last_actions = row['previous_actions']
            for i in range(window_size):
                X.at[index, 'previous_action_{}'.format(i)] = last_actions[i]

        return X
