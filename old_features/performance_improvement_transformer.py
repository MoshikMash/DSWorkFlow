import pandas as pd
from sklearn.base import TransformerMixin
import json
from copy import deepcopy


class PerformanceImprovement(TransformerMixin):
    def __init__(self):
        self.configuration_data = json.loads(open("configuration.json", "r").read())
        self.metrics = ['recall', 'accuracy', 'auc', 'confusion_matrix']
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

    def add_improvement_column(self, X, metric):
        flag = False
        check_flag = True
        improvement = list()
        best_scores = list()
        best_metric = 0.0001
        if metric == 'confusion_matrix':
            best_metric = 1
        for index, row in X.iterrows():
            if best_metric == 0.0001 or best_metric == 1:
                improvement.append(0)
            elif not (row['metric_type'] == 'test' and row['score'] == metric):
                if flag:
                    improvement.append(1)
                else:
                    improvement.append(0)
            elif flag:
                improvement.append(1)
                check_flag = True
            else:
                improvement.append(0)
                check_flag = True
            best_scores.append(best_metric)
            if row['metric_type'] == 'test' and row['score'] == metric and check_flag:
                if metric != 'confusion_matrix':
                    if row['metric'] >= best_metric:
                        best_metric = row['metric']
                        flag = True
                        check_flag = False
                    else:
                        flag = False
                elif row['cm_content'] != -1:
                    tp = int(list(row['cm_content'])[3])
                    if tp >= best_metric:
                        best_metric = tp
                        flag = True
                        check_flag = False
                    else:
                        flag = False
        X['impr_{}'.format(metric)] = improvement
        X['best_score_{}'.format(metric)] = best_scores

        return X

    def append_features(self, X):
        for metric in self.metrics:
            X = self.add_improvement_column(X, metric)

        return X
