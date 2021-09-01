import pandas as pd
from sklearn.base import TransformerMixin
import json
import copy


class TrainEvalLoop(TransformerMixin):
    def __init__(self):
        self.features = ['train_eval_flag']
        self.enc = False

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.features]

    def get_feature_names(self):
        return self.features

    def append_features(self, X):
        train_eval_loop_flag = False
        train_eval_loop_flags_list = list()
        for index, row in X.iterrows():
            if 'train' in row['action'] or 'eval' in row['action']:
                train_eval_loop_flag = True
            elif train_eval_loop_flag and 'OTHER' in row['action']:
                train_eval_loop_flag = True
            else:
                train_eval_loop_flag = False
            train_eval_loop_flags_list.append(train_eval_loop_flag)
        X['train_eval_flag'] = train_eval_loop_flags_list
