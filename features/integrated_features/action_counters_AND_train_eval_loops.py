import json
from sklearn.base import TransformerMixin


class IntegratedActionCounterANDTrainEvalLoopCounter(TransformerMixin):
    """
    this feature divide the values of
    counters_with_time_window_size_action_changed_between_train_eval_loops_transformer
    by the value of number_of_train_eval_loops
    if the values of number_of_train_eval_loops is zero so the feature value will be zero as well
    """

    def __init__(self):
        self.configuration_data = json.loads(open("configuration.json", "r").read())
        self.possible_actions = self.configuration_data['possible_actions']
        self.column_names = []
        for action in self.possible_actions:
            self.column_names.append('#{}/#train_eval_loops'.format(action))

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.column_names]

    def get_feature_names(self):
        return self.column_names

    @staticmethod
    def append_feature_for_action(X, action):
        feature_values = []
        for index, row in X.iterrows():
            if row['number_of_train_eval_loops_in_time_window'] == 0:
                feature_values.append(0)
            else:
                feature_values.append(
                    row['{}_counter_with_time_window_size_action_was_changed_between_train_eval_loops'.format(action)] /
                    row['number_of_train_eval_loops_in_time_window'])

        X['#{}/#train_eval_loops'.format(action)] = feature_values

        return X

    def append_features(self, X):
        for action in self.possible_actions:
            X = self.append_feature_for_action(X, action)
