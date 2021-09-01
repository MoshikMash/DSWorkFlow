import pandas as pd
from sklearn.base import TransformerMixin
import json
from sklearn.preprocessing import OneHotEncoder


class BestMLAlgo(TransformerMixin):
    def __init__(self):
        self.configuration_data = json.loads(open("configuration.json", "r").read())
        self.metrics = self.configuration_data['performance_metrics']
        self.column_names = list()
        self.column_names_groups = list()
        for metric in self.metrics:
            self.column_names.append('best_ml_algo_{}'.format(metric))

        # for metric in self.metrics:
        #     self.column_names.append('best_score_{}'.format(metric))

        self.enc = None

    def fit(self, X):
        self.enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.enc.fit(X[self.column_names])
        return self

    def transform(self, X):
        return self.enc.transform(X[self.column_names])

    def get_feature_names(self):
        return self.enc.get_feature_names(self.column_names)

    def check_ml_algo(self, code_line):
        if 'Logistic' in code_line:
            return 'LR'
        elif 'Random' in code_line:
            return 'RF'
        else:
            return 'OTHER'

    def add_best_ml_algo_column(self, X, metric):
        improvement = list()
        best_scores = list()
        best_metric = 0.0001
        ml_algo = '-'
        best_algo = '-'
        if metric == 'confusion_matrix':
            best_metric = 1
        for index, row in X.iterrows():
            if 'train_model' in row['code']:
                ml_algo = self.check_ml_algo(row['code'])
            if best_metric == 0.0001 or best_metric == 1:
                improvement.append(best_algo)
            elif not (row['metric_type'] == 'test' and row['score'] == metric):
                improvement.append(best_algo)
            else:
                improvement.append(best_algo)
            if row['metric_type'] == 'test' and row['score'] == metric:
                if metric != 'confusion_matrix':
                    if row['metric'] >= best_metric:
                        best_algo = ml_algo
                        best_metric = row['metric']
                elif row['cm_content'] != -1:
                    tp = int(list(row['cm_content'])[3])
                    if tp >= best_metric:
                        best_metric = tp

        X['best_ml_algo_{}'.format(metric)] = improvement

        # best_ml_algo_list = list()
        # for index, row in X.iterrows():
        #     if ('train' not in row['previous_action_0']) and ('hyper' not in row['previous_action_0']) and (
        #             'eval' not in row['previous_action_0']):
        #         best_ml_algo_list.append('-')
        #     else:
        #         best_ml_algo_list.append(row['best_ml_algo_{}'.format(metric)])
        #
        # X['best_ml_algo_{}'.format(metric)] = best_ml_algo_list

        return X

    def append_features(self, X):
        for metric in self.metrics:
            X = self.add_best_ml_algo_column(X, metric)

        return X
