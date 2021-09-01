from sklearn.base import TransformerMixin
import json
from copy import deepcopy


class ActionChangedBetweenTrainEvalLoops(TransformerMixin):
    def __init__(self):
        self.configuration_data = json.loads(open("configuration.json", "r").read())
        self.possible_actions = self.configuration_data['possible_actions']
        # self.possible_actions = ['feature_sel', 'FE_binning', 'feature_scaling']
        self.counter_columns = list()
        actions = self.configuration_data['possible_actions']
        # actions = ['feature_sel', 'FE_binning', 'feature_scaling', 'visualize']
        for action in actions:
            self.counter_columns.append('{}_action_was_changed_between_train_eval_loops'.format(action))

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.counter_columns]

    def get_feature_names(self):
        return self.counter_columns

    def append_features(self, X):
        labels = X['label'].values
        codes = X['code'].values
        features = X['features'].values
        cell_numbers = X['cell_number'].values
        different_actions_flag_dictionary = {key: False for key in self.possible_actions}
        last_code_per_action = {key: '-' for key in self.possible_actions}
        last_feature_list_per_action = {key: '-' for key in self.possible_actions}
        flag_model_evaluation_dictionary = {key: True for key in self.possible_actions}

        last_code_per_action_list_last_loop = {key: list() for key in self.possible_actions}
        last_code_per_action_list_current_loops = {key: list() for key in self.possible_actions}
        last_loop_and_current_loop_was_replaced = True

        sum1 = 0
        # the first element's value should be empty dictionary
        triggers = list()

        for i, label in enumerate(labels):
            if cell_numbers[i] != -1:
                flag_model_evaluation_dictionary = {key: True for key in self.possible_actions}
                if not last_loop_and_current_loop_was_replaced:
                    last_code_per_action_list_last_loop = deepcopy(last_code_per_action_list_current_loops)
                    last_code_per_action_list_current_loops = {key: list() for key in self.possible_actions}
                    different_actions_flag_dictionary = {key: False for key in self.possible_actions}
                    last_loop_and_current_loop_was_replaced = True
            if label in self.possible_actions:
                if label != 'FE_binning':
                    if flag_model_evaluation_dictionary[label]:
                        if codes[i] != last_code_per_action[label] or\
                                features[i] != last_feature_list_per_action[label]:
                            different_actions_flag_dictionary[label] = True
                            last_code_per_action[label] = deepcopy(codes[i])
                            last_feature_list_per_action[label] = deepcopy(features[i])
                        else:
                            different_actions_flag_dictionary[label] = False
                        flag_model_evaluation_dictionary[label] = False
                else:
                    last_loop_and_current_loop_was_replaced = False
                    if codes[i] not in last_code_per_action_list_last_loop[label]:
                        different_actions_flag_dictionary[label] = True

                    last_code_per_action_list_current_loops[label].append(deepcopy(codes[i]))

                    flag_model_evaluation_dictionary[label] = False

            triggers.append(deepcopy(different_actions_flag_dictionary))

        X['trigger_action_was_changed_between_train_eval_loops'] = triggers

        # create column for each trigger
        for action in different_actions_flag_dictionary:
            l = []
            for index, row in X.iterrows():
                l.append(row['trigger_action_was_changed_between_train_eval_loops'][action])
            X['{}_action_was_changed_between_train_eval_loops'.format(action)] = l

        return X
