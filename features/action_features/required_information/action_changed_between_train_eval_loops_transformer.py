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
        actions = X['action'].values
        codes = X['code'].values
        features = X['features'].values
        train_eval_flags = X['train_eval_flag'].values
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

        for i, action in enumerate(actions):
            if train_eval_flags[i]:
                flag_model_evaluation_dictionary = {key: True for key in self.possible_actions}
                if not last_loop_and_current_loop_was_replaced:
                    last_code_per_action_list_last_loop = deepcopy(last_code_per_action_list_current_loops)
                    last_code_per_action_list_current_loops = {key: list() for key in self.possible_actions}
                    different_actions_flag_dictionary = {key: False for key in self.possible_actions}
                    last_loop_and_current_loop_was_replaced = True
            if action in self.possible_actions:
                if action != 'FE_binning':
                    if flag_model_evaluation_dictionary[action]:
                        if codes[i] != last_code_per_action[action] or \
                                features[i] != last_feature_list_per_action[action]:
                            different_actions_flag_dictionary[action] = True
                            last_code_per_action[action] = deepcopy(codes[i])
                            last_feature_list_per_action[action] = deepcopy(features[i])
                        else:
                            different_actions_flag_dictionary[action] = False
                        flag_model_evaluation_dictionary[action] = False
                else:
                    last_loop_and_current_loop_was_replaced = False
                    if codes[i] not in last_code_per_action_list_last_loop[action]:
                        different_actions_flag_dictionary[action] = True

                    last_code_per_action_list_current_loops[action].append(deepcopy(codes[i]))

                    flag_model_evaluation_dictionary[action] = False

            triggers.append(deepcopy(different_actions_flag_dictionary))

        X['trigger_action_was_changed_between_train_eval_loops'] = triggers

        # create column for each trigger
        for action in different_actions_flag_dictionary:
            final_flags = []
            for index, row in X.iterrows():
                final_flags.append(row['trigger_action_was_changed_between_train_eval_loops'][action])
            X['{}_action_was_changed_between_train_eval_loops'.format(action)] = final_flags

        # # assign True only when the the values was changed and not till the end of the train-eval loop
        # for action in different_actions_flag_dictionary:
        #     updated_values = list()
        #     for index, row in X.iterrows():
        #         if row['action'] != action:
        #             updated_values.append(False)
        #         else:
        #             updated_values.append(row['{}_action_was_changed_between_train_eval_loops'.format(action)])
        #     X['{}_action_was_changed_between_train_eval_loops'.format(action)] = updated_values
