import pandas as pd
from sklearn.base import TransformerMixin
import json
from copy import deepcopy


class BestScoreAndImprovement(TransformerMixin):
    def __init__(self):
        self.metrics = ['precision', 'recall', 'accuracy', 'auc']
        self.column_names = list()
        for metric in self.metrics:
            self.column_names.append('impr_{}'.format(metric))

        # for metric in self.metrics:
        #     self.column_names.append('best_score_{}'.format(metric))

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.column_names]

    def get_feature_names(self):
        return self.column_names

    @staticmethod
    def append_best_score_and_improvement_column(X, metric):
        best_scores = list()
        improvements = list()
        best_score_util_now = 0.0001
        for index, row in X.iterrows():
            if row['metric_type'] == 'test' and row['metric'] == metric:
                best_score_util_now = row['score']
                if row['score'] > best_score_util_now + 0.1:
                    improvements.append(1)
                else:
                    improvements.append(0)
            else:
                improvements.append(-1)
            best_scores.append(best_score_util_now)

        X['score_improved_{}'.format(metric)] = improvements
        X['best_score_{}'.format(metric)] = best_scores

        return X

    def append_features(self, X):
        for metric in self.metrics:
            X = self.append_best_score_and_improvement_column(X, metric)
