import pandas as pd
from sklearn.base import TransformerMixin
import json
from sklearn.preprocessing import OneHotEncoder


class LastPerformanceImprovement(TransformerMixin):
    def __init__(self):
        self.configuration_data = json.loads(open("configuration.json", "r").read())
        self.metrics = self.configuration_data['performance_metrics']
        self.column_names = list()
        self.column_names_groups = list()
        for metric in self.metrics:
            self.column_names.append('impr_last_{}'.format(metric))
            self.column_names_groups.append('impr_last_groups_{}'.format(metric))

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

    def add_last_improved_column(self, X, metric):
        flag = False
        check_flag = True
        improvement = list()
        best_scores = list()
        best_metric = 0.0001
        counter = 0
        if metric == 'confusion_matrix':
            best_metric = 1
        for index, row in X.iterrows():
            if best_metric == 0.0001 or best_metric == 1:
                improvement.append(counter)
            elif not (row['metric_type'] == 'test' and row['score'] == metric):
                improvement.append(counter)
            else:
                check_flag = True
                improvement.append(counter)
            best_scores.append(best_metric)
            if row['metric_type'] == 'test' and row['score'] == metric and check_flag:
                if metric != 'confusion_matrix':
                    if row['metric'] >= best_metric:
                        counter = 0
                        best_metric = row['metric']
                        flag = True
                        check_flag = False
                    else:
                        flag = False
                        counter += 1
                elif row['cm_content'] != -1:
                    tp = int(list(row['cm_content'])[3])
                    if tp >= best_metric:
                        best_metric = tp
                        flag = True
                        check_flag = False
                        counter = 0
                    else:
                        flag = False
                        counter += 1
        X['impr_last_{}'.format(metric)] = improvement

        return X

    def append_features(self, X):
        for metric in self.metrics:
            X = self.add_last_improved_column(X, metric)

        performance_counters_bins = self.configuration_data['performance_counters_bins']
        for metric in self.metrics:
            if performance_counters_bins:
                X['impr_last_groups_{}'.format(metric)] = pd.cut(x=X['impr_last_{}'.format(metric)],
                                                                    bins=performance_counters_bins)
            else:
                X['impr_last_groups_{}'.format(metric)] = X['impr_last_{}'.format(metric)]

        return X
