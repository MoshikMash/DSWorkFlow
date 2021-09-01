from sklearn.base import TransformerMixin
import json
from copy import deepcopy


class LastTimeActionPerformed(TransformerMixin):
    def __init__(self):
        self.configuration_data = json.loads(open("configuration.json", "r").read())
        self.possible_actions = self.configuration_data['possible_actions']
        self.possible_actions = self.possible_actions
        self.last_time_performed = list()
        actions = self.possible_actions
        for action in actions:
            self.last_time_performed.append('{}_last_time_performed'.format(action))

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.last_time_performed]

    def get_feature_names(self):
        return self.last_time_performed

    def append_features(self, X):
        labels = X['label'].values
        last_time_action_performed = {key: 0 for key in self.possible_actions}
        # the first element's value should be empty dictionary
        triggers = [deepcopy(last_time_action_performed)]

        for label in labels:
            if label in last_time_action_performed:
                last_time_action_performed[label] = 0
                for label2 in last_time_action_performed:
                    if label2 != label:
                        last_time_action_performed[label2] += 1

            triggers.append(deepcopy(last_time_action_performed))

        # drop the last dictionary
        triggers.pop()

        X['trigger_last_time_performed'] = triggers

        # create column for each trigger
        for action in last_time_action_performed:
            l = []
            for index, row in X.iterrows():
                l.append(row['trigger_last_time_performed'][action])
            X['{}_last_time_performed'.format(action)] = l

        return X
