from sklearn.base import TransformerMixin
import numpy as np


class NumberOfTrainEvalLoops(TransformerMixin):
    def __init__(self):
        self.features = ['number_of_train_eval_loops']

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.features]

    def get_feature_names(self):
        return self.features

    def append_features(self, X):
        cell_numbers = list(X['cell_number'])
        number_of_train_eval_loops = list()
        for i, cell_num in enumerate(cell_numbers):
            if i == 0:
                number_of_train_eval_loops.append(0)
            elif cell_num == -1:
                number_of_train_eval_loops.append(np.max(cell_numbers[:i]) + 1)
            else:
                number_of_train_eval_loops.append(cell_num + 1)

        X['number_of_train_eval_loops'] = number_of_train_eval_loops
        return X
