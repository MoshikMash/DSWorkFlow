from sklearn.base import TransformerMixin
import numpy as np
from copy import deepcopy
import json
from sklearn.preprocessing import OneHotEncoder


class NumberOfModifiedFeatures(TransformerMixin):
    def __init__(self):
        self.configuration_data = json.loads(open("configuration.json", "r").read())
        self.possible_actions = self.configuration_data['possible_actions']
        actions_to_remove_from_dict = ['train', 'eval', 'splitting', 'visualize']
        eval_actions = ['eval_cm', 'eval_precision', 'eval_recall', 'eval_accuracy', 'eval_auc', 'eval_f1',
                        'eval_roc_curve', 'eval_get_predictions', 'get_predictions_proba']
        actions_to_remove_from_dict.extend(eval_actions)
        for action in actions_to_remove_from_dict:
            if action in self.possible_actions:
                self.possible_actions.remove(action)

        self.num_of_modified_features_columns = list()
        for action in self.possible_actions:
            self.num_of_modified_features_columns.append('{}_trigger_num_of_modified_features'.format(action))

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.num_of_modified_features_columns]

    def get_feature_names(self):
        return self.num_of_modified_features_columns

    def append_features(self, X):
        num_of_modified_features = list(X['num of modified features'].values)
        num_of_modified_features = [x if x != '' else 0 for x in num_of_modified_features]
        labels = X['label'].values
        num_of_modified_features_dict = {key: 0 for key in self.possible_actions}
        triggers = [deepcopy(num_of_modified_features_dict)]

        for i, label in enumerate(labels):
            if label in self.possible_actions:
                num_of_modified_features_dict[label] = num_of_modified_features[i]
            triggers.append(deepcopy(num_of_modified_features_dict))
        # drop the last dictionary
        triggers.pop()

        X['trigger_num_of_modified_features'] = triggers

        for action in num_of_modified_features_dict:
            l = []
            for index, row in X.iterrows():
                l.append(row['trigger_num_of_modified_features'][action])
            X['{}_trigger_num_of_modified_features'.format(action)] = l
            X['{}_trigger_num_of_modified_features'.format(action)] = X[
                '{}_trigger_num_of_modified_features'.format(action)].astype(int)

        return X
