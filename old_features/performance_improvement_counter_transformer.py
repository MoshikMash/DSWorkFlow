import pandas as pd
from sklearn.base import TransformerMixin
import json
from copy import deepcopy
from sklearn.preprocessing import OneHotEncoder


class PerformanceImprovementCounter(TransformerMixin):
    def __init__(self):
        self.configuration_data = json.loads(open("configuration.json", "r").read())
        self.metrics = self.configuration_data['performance_metrics']
        self.column_names = list()
        self.column_names_groups = list()
        for metric in self.metrics:
            self.column_names.append('impr_counter_{}'.format(metric))
            self.column_names_groups.append('impr_counter_groups_{}'.format(metric))

        # for metric in self.metrics:
        #     self.column_names.append('best_score_{}'.format(metric))

        self.enc = None

    def fit(self, X):
        self.enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.enc.fit(X[self.column_names_groups])
        return self

    def transform(self, X):
        return self.enc.transform(X[self.column_names_groups])

    def get_feature_names(self):
        return self.enc.get_feature_names(self.column_names_groups)

    def add_improvement_counter_column(self, X, metric):
        flag = False
        check_flag = True
        improvement = list()
        best_scores = list()
        best_metric = 0.0001
        counter = 0
        if 'confusion_matrix' in metric:
            best_metric = 1
        for index, row in X.iterrows():
            if best_metric == 0.0001 or best_metric == 1:
                improvement.append(counter)
            elif not (row['metric_type'] == 'test' and row['score'] in metric):
                improvement.append(counter)
            else:
                check_flag = True
                improvement.append(counter)
            best_scores.append(best_metric)
            if row['metric_type'] == 'test' and row['score'] in metric and check_flag:
                if 'confusion_matrix' not in metric:
                    if row['metric'] >= best_metric:
                        counter += 1
                        best_metric = row['metric']
                        flag = True
                        check_flag = False
                    else:
                        flag = False
                elif row['cm_content'] != -1:
                    cm_index = int(metric[-1])
                    if isinstance(row['cm_content'], list):
                        tp = int(float(row['cm_content'][cm_index]))
                    else:
                        tp = int(float(row['cm_content'].strip('][').split(', ')[cm_index]))
                    if tp >= best_metric:
                        best_metric = tp
                        flag = True
                        check_flag = False
                        counter += 1
                    else:
                        flag = False
        X['impr_counter_{}'.format(metric)] = improvement

        return X

    def append_features(self, X):
        for metric in self.metrics:
            X = self.add_improvement_counter_column(X, metric)

        counters_bins = self.configuration_data['performance_counters_bins']
        for metric in self.metrics:
            if counters_bins:
                X['impr_counter_groups_{}'.format(metric)] = pd.cut(x=X['impr_counter_{}'.format(metric)],
                                                                    bins=counters_bins)
            else:
                X['impr_counter_groups_{}'.format(metric)] = X['impr_counter_{}'.format(metric)]

        return X
