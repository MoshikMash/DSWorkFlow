from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np
from copy import deepcopy
import json
from sklearn.preprocessing import OneHotEncoder
from actionPrediction.features.number_of_modified_features_transformer import NumberOfModifiedFeatures


class NumberOfModifiedFeaturesGroups(TransformerMixin):
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
        self.num_of_modified_features_groups_columns = list()
        for action in self.possible_actions:
            self.num_of_modified_features_columns.append('{}_trigger_num_of_modified_features'.format(action))
            self.num_of_modified_features_groups_columns.append('{}_groups_number_of_modified_features'.format(action))

        self.enc = None

    def fit(self, X):
        self.enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.enc.fit(X[self.num_of_modified_features_groups_columns])
        return self

    def transform(self, X):
        return self.enc.transform(X[self.num_of_modified_features_groups_columns])

    def get_feature_names(self):
        return self.enc.get_feature_names(self.num_of_modified_features_groups_columns)

    def append_features(self, X):
        number_of_modified_features_obj = NumberOfModifiedFeatures()
        number_of_modified_features_obj.append_features(X)

        bins = self.configuration_data['number_of_modified_features_bins']
        for action in self.possible_actions:
            X['{}_groups_number_of_modified_features'.format(action)] = pd.cut(
                x=X['{}_trigger_num_of_modified_features'.format(action)], bins=bins)

        return X
