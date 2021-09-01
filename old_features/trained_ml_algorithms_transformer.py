import pandas as pd
from sklearn.base import TransformerMixin
import json
import copy


class TrainedMLalgorithms(TransformerMixin):
    def __init__(self):
        self.configuration_data = json.loads(open("configuration.json", "r").read())
        self.ml_algorithms = ['LogisticRegression', 'RandomForest', 'DecisionTree', 'KNN', 'LDA', 'MultinomialNB',
                              'GaussianNB']
        self.features = list()
        for ml_algo in self.ml_algorithms:
            self.features.append('{}_trained'.format(ml_algo))

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.features]

    def get_feature_names(self):
        return self.features

    def append_features(self, X):
        ml_dict = dict()
        ml_dict_boolean = dict()
        for ml_algo in self.ml_algorithms:
            ml_dict[ml_algo] = list([0])
            ml_dict_boolean[ml_algo] = 0
        for index, row in X.iterrows():
            cycle_num = row['cell_number']
            if cycle_num != -1:
                for ml_algo in ml_dict:
                    if ml_algo in row['code']:
                        ml_dict_boolean[ml_algo] = 1
                    ml_dict[ml_algo].append(ml_dict_boolean[ml_algo])
            else:
                for ml_algo in ml_dict:
                    ml_dict_boolean[ml_algo] = 0
                    ml_dict[ml_algo].append(ml_dict_boolean[ml_algo])

        for ml_algo in ml_dict:
            ml_dict[ml_algo].pop()
            X['{}_trained'.format(ml_algo)] = ml_dict[ml_algo]

        return X
