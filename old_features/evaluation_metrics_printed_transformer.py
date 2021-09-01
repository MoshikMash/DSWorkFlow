import pandas as pd
from sklearn.base import TransformerMixin
import json
import copy


class EvaluationMetricsPrinted(TransformerMixin):
    def __init__(self):
        self.configuration_data = json.loads(open("configuration.json", "r").read())
        self.metrics = ['recall', 'precision', 'auc', 'roc', 'eval_cm', 'accuracy']
        self.features = list()
        for metric in self.metrics:
            self.features.append('{}_printed'.format(metric))

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.features]

    def get_feature_names(self):
        return self.features

    def append_features(self, X):
        metrics_dict = dict()
        metric_dict_boolean = dict()
        for metric in self.metrics:
            metrics_dict[metric] = list([0])
            metric_dict_boolean[metric] = 0
        for index, row in X.iterrows():
            cycle_num = row['cell_number']
            if cycle_num != -1:
                for metric in metrics_dict:
                    if metric in row['code']:
                        metric_dict_boolean[metric] = 1
                    metrics_dict[metric].append(metric_dict_boolean[metric])
            else:
                for metric in metrics_dict:
                    metric_dict_boolean[metric] = 0
                    metrics_dict[metric].append(metric_dict_boolean[metric])

        for metric in metrics_dict:
            metrics_dict[metric].pop()
            X['{}_printed'.format(metric)] = metrics_dict[metric]

        return X
