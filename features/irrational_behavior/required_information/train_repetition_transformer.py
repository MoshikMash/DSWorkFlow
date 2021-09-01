import json
from sklearn.base import TransformerMixin
from copy import deepcopy


class TrainRepetition(TransformerMixin):
    """
    This feature counts the times that the data scientist train and evaluate a model/s,
    than executed some actions without any changes and then train and evaluate the same model/s again

    Notes:
        1. actions required_information are required for that feature as well
        2. train_eval_loop required_information are required fir that feature
    """

    def __init__(self):
        self.column_names = ['train_repetition_flag']

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.column_names]

    def get_feature_names(self):
        return self.column_names

    def append_features(self, X):
        repetition_models = []
        last_actions_status = None
        start_of_train_eval_loop = True
        for index, row in X.iterrows():
            if row['train_eval_flag'] == 1:
                if start_of_train_eval_loop:
                    if last_actions_status == current_actions_status:
                        repetition_models.append(1)
                    else:
                        repetition_models.append(0)
                    last_actions_status = deepcopy(current_actions_status)
                else:
                    repetition_models.append(0)

                start_of_train_eval_loop = False
            else:
                current_actions_status = deepcopy(row['trigger_action_was_changed_between_train_eval_loops'])
                del current_actions_status['train']
                del current_actions_status['eval']
                del current_actions_status['OTHER']
                del current_actions_status['visualize']
                repetition_models.append(0)

                start_of_train_eval_loop = True

        X['train_repetition_flag'] = repetition_models
