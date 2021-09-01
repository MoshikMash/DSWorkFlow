from sklearn.base import TransformerMixin
import json
from copy import deepcopy


class LastTimeActionPerformedWhenChanged(TransformerMixin):
    def __init__(self):
        self.configuration_data = json.loads(open("configuration.json", "r").read())
        self.possible_actions = self.configuration_data['possible_actions']
        features_to_not_include = [
            "train"
            "eval",
            "hyperparameter_tuning",
            "eval",
            'eval_roc_curve'
        ]
        self.possible_actions = [x for x in self.possible_actions if x not in features_to_not_include]

        self.possible_actions = self.possible_actions
        self.last_time_performed = list()
        actions = self.possible_actions
        for action in actions:
            self.last_time_performed.append('{}_last_time_performed_when_changed'.format(action))

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.last_time_performed]

    def get_feature_names(self):
        return self.last_time_performed

    def append_features(self, X):
        labels = X['label'].values
        cell_numbers = X['cell_number'].values
        last_time_action_performed = {key: 0 for key in self.possible_actions}
        action_was_changed_between_train_eval_loops_dict = {key: None for key in self.possible_actions}
        for key in action_was_changed_between_train_eval_loops_dict:
            action_was_changed_between_train_eval_loops_dict[
                key] = X['{}_action_was_changed_between_train_eval_loops'.format(key)].values

        triggers = list()

        visited_train_eval_loop_flag = False
        for i, label in enumerate(labels):
            if cell_numbers[i] != -1 and not visited_train_eval_loop_flag:
                for label2 in last_time_action_performed:
                    last_time_action_performed[label2] += 1
                    visited_train_eval_loop_flag = True
            elif label in last_time_action_performed and cell_numbers[i] == -1:
                visited_train_eval_loop_flag = False
                if action_was_changed_between_train_eval_loops_dict[label][i]:
                    last_time_action_performed[label] = 0

            triggers.append(deepcopy(last_time_action_performed))

        X['trigger_last_time_performed_when_changed'] = triggers

        # create column for each trigger
        for action in last_time_action_performed:
            l = []
            for index, row in X.iterrows():
                l.append(row['trigger_last_time_performed_when_changed'][action])
            X['{}_last_time_performed_when_changed'.format(action)] = l

        return X
