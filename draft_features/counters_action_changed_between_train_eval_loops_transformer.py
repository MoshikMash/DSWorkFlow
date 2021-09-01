from sklearn.base import TransformerMixin
import json
from copy import deepcopy


class CountersActionChangedBetweenTrainEvalLoops(TransformerMixin):
    def __init__(self):
        self.configuration_data = json.loads(open("configuration.json", "r").read())
        self.possible_actions = self.configuration_data['possible_actions']
        self.counter_columns = list()
        actions = self.possible_actions
        for action in actions:
            self.counter_columns.append('{}_counter_action_was_changed_between_train_eval_loops'.format(action))

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.counter_columns]

    def get_feature_names(self):
        return self.counter_columns

    def append_features(self, X):
        counters_dictionary = {key: 0 for key in self.possible_actions}
        train_eval_visited_flags = {key: True for key in self.possible_actions}

        # the first element's value should be empty dictionary
        triggers = list()

        for index, row in X.iterrows():
            if row['train_eval_flag']:
                train_eval_visited_flags = {key: True for key in self.possible_actions}
            if row['action'] in counters_dictionary and train_eval_visited_flags[row['action']]:
                if row['{}_action_was_changed_between_train_eval_loops'.format(row['action'])]:
                    counters_dictionary[row['action']] += 1
                    train_eval_visited_flags[row['action']] = False
            triggers.append(deepcopy(counters_dictionary))

        X['trigger_counters_action_was_changed_between_train_eval_loops'] = triggers

        # create column for each trigger
        for action in counters_dictionary:
            final_counters = []
            for index, row in X.iterrows():
                final_counters.append(row['trigger_counters_action_was_changed_between_train_eval_loops'][action])
            X['{}_counter_action_was_changed_between_train_eval_loops'.format(action)] = deepcopy(final_counters)
