from sklearn.base import TransformerMixin
import json
from copy import deepcopy
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class LastTimeActionPerformedGroups(TransformerMixin):
    def __init__(self):
        self.configuration_data = json.loads(open("configuration.json", "r").read())
        self.possible_actions = self.configuration_data['possible_actions']
        self.possible_actions = ['FE_binning', 'feature_sel']
        self.last_time_performed = list()
        self.last_time_performed_groups = list()
        actions = self.possible_actions
        for action in actions:
            self.last_time_performed.append('{}_last_time_performed'.format(action))
            self.last_time_performed_groups.append('{}_groups_last_time_performed'.format(action))

        self.enc = None

    def fit(self, X):
        self.enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.enc.fit(X[self.last_time_performed_groups])
        return self

    def transform(self, X):
        return self.enc.transform(X[self.last_time_performed_groups])

    def get_feature_names(self):
        return self.enc.get_feature_names(self.last_time_performed_groups)

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

        actions_bins = self.configuration_data['counters_groups_bins']
        for action in self.possible_actions:
            if 'eval' in action:
                action_bins_key = 'eval'
            else:
                action_bins_key = action
            bins = actions_bins[action_bins_key]
            X['{}_groups_last_time_performed'.format(action)] = pd.cut(x=X['{}_last_time_performed'.format(action)],
                                                                       bins=bins)

        return X
