from sklearn.base import TransformerMixin
import numpy as np


class CounterTrainEvalLoops(TransformerMixin):
    def __init__(self):
        self.features = ['number_of_train_eval_loops']

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.features]

    def get_feature_names(self):
        return self.features

    def append_features(self, X):
        number_of_train_eval_loops = list()
        number_of_train_eval_loops_counter = 0
        train_action_was_visited = False
        for index, row in X.iterrows():
            if row['train_eval_flag']:
                if not train_action_was_visited:
                    number_of_train_eval_loops_counter += 1
                    train_action_was_visited = True
            elif not row['train_eval_flag']:
                train_action_was_visited = False
            number_of_train_eval_loops.append(number_of_train_eval_loops_counter)
        X['number_of_train_eval_loops'] = number_of_train_eval_loops
