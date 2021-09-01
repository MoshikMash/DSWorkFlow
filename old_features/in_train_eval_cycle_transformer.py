import pandas as pd
from sklearn.base import TransformerMixin
import json
import copy


class InTrainEvalCycle(TransformerMixin):
    def __init__(self):
        self.configuration_data = json.loads(open("configuration.json", "r").read())
        self.features = ['train_eval_flag']
        self.enc = False

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.features]

    def get_feature_names(self):
        return self.features

    def append_features(self, X):
        flag = []
        last_flag = 0
        for index, row in X.iterrows():
            cycle_num = row['cell_number']
            if cycle_num != -1:
                if last_flag == 1:
                    flag.append(1)
                else:
                    flag.append(0)
                last_flag = 1
            else:
                if last_flag == 0:
                    flag.append(0)
                else:
                    flag.append(1)
                last_flag = 0
        X['train_eval_flag'] = flag

        return X