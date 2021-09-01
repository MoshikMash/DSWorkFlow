from sklearn.base import TransformerMixin
import json
from copy import deepcopy


class CountersBetweenTrainEvalLoops(TransformerMixin):
    def __init__(self):
        self.configuration_data = json.loads(open("configuration.json", "r").read())
        self.possible_actions = self.configuration_data['possible_actions']
        self.counter_columns = list()
        actions = self.configuration_data['possible_actions']
        for action in actions:
            self.counter_columns.append('{}_counter_between_train_eval_loops'.format(action))

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.counter_columns]

    def get_feature_names(self):
        return self.counter_columns

    def append_features(self, X):
        labels = X['label'].values
        cell_numbers = X['cell_number'].values
        counters_dictionary = {key: 0 for key in self.possible_actions}
        flag_model_evaluation_dictionary = {key: True for key in self.possible_actions}

        sum1 = 0
        # the first element's value should be empty dictionary
        triggers = [deepcopy(counters_dictionary)]

        for i, label in enumerate(labels):
            if cell_numbers[i] != -1:
                flag_model_evaluation_dictionary = {key: True for key in self.possible_actions}
            if label in counters_dictionary:
                if flag_model_evaluation_dictionary[label]:
                    counters_dictionary[label] += 1
                    sum1 += 1
                    flag_model_evaluation_dictionary[label] = False
            triggers.append(deepcopy(counters_dictionary))

        # drop the last dictionary
        triggers.pop()

        for i, label in enumerate(labels):
            if label in counters_dictionary:
                if i < len(labels) - 1:
                    if triggers[i][label] > triggers[i+1][label]:
                        triggers[i][label] += 1
                else:
                    triggers[i][label] += 1

        X['trigger_counter_between_train_eval_loops'] = triggers

        # create column for each trigger
        for action in counters_dictionary:
            l = []
            for index, row in X.iterrows():
                l.append(row['trigger_counter_between_train_eval_loops'][action])
            X['{}_counter_between_train_eval_loops'.format(action)] = l

        return X
