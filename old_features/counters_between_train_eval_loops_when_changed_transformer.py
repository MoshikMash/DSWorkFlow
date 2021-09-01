from sklearn.base import TransformerMixin
import json
from copy import deepcopy


class CountersBetweenTrainEvalLoopsWhenChanged(TransformerMixin):
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

        self.counter_columns = list()
        actions = self.possible_actions
        for action in actions:
            self.counter_columns.append('{}_counter_between_train_eval_loops_when_changed'.format(action))

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

        action_was_changed_between_train_eval_loops_dict = {key: None for key in self.possible_actions}
        for key in action_was_changed_between_train_eval_loops_dict:
            action_was_changed_between_train_eval_loops_dict[
                key] = X['{}_action_was_changed_between_train_eval_loops'.format(key)].values

        sum1 = 0
        # the first element's value should be empty dictionary
        triggers = list()

        for i, label in enumerate(labels):
            if cell_numbers[i] != -1:
                flag_model_evaluation_dictionary = {key: True for key in self.possible_actions}
            if label in counters_dictionary:
                if flag_model_evaluation_dictionary[label]:
                    # if label in ['feature_sel', 'FE_binning', 'feature_scaling']:
                    if action_was_changed_between_train_eval_loops_dict[label][i]:
                        counters_dictionary[label] += 1
                        sum1 += 1
                        flag_model_evaluation_dictionary[label] = False
                    # else:
                    #     counters_dictionary[label] += 1
                    #     sum1 += 1
                    #     flag_model_evaluation_dictionary[label] = False
            triggers.append(deepcopy(counters_dictionary))

        X['trigger_counter_between_train_eval_loops_when_changed'] = triggers

        # create column for each trigger
        for action in counters_dictionary:
            l = []
            for index, row in X.iterrows():
                l.append(row['trigger_counter_between_train_eval_loops_when_changed'][action])
            X['{}_counter_between_train_eval_loops_when_changed'.format(action)] = l

        return X
