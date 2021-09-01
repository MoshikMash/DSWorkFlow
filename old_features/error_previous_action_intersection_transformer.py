from sklearn.base import TransformerMixin
import json
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class ErrorWithPreviousAction(TransformerMixin):
    def __init__(self):
        self.configuration_data = json.loads(open("configuration.json", "r").read())
        self.error_dict = pickle.load(open("features/dicts/error_dict.p", 'rb'))

        self.features = None
        self.enc = False

    def fit(self, X):
        self.enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.enc.fit(X[['error_previous_action_intersection']])
        self.features = self.enc.get_feature_names(list(set(X[['error_previous_action_intersection']])))
        return self

    def transform(self, X):
        return self.enc.transform(X[['error_previous_action_intersection']])

    def get_feature_names(self):
        return self.features

    def append_error_feature(self, X):
        errors = list([0])
        for index, row in X.iterrows():
            s_code = row['session_code']
            code_line = row['code']
            error_dic = self.error_dict[s_code]
            error = error_dic.get(code_line, 0)
            if error == "error":
                errors.append(1)
            else:
                errors.append(0)
        errors.pop()
        X['error'] = errors
        return X

    def append_features(self, X):
        self.append_error_feature(X)
        X['error_previous_action_intersection'] = ['{}_{}'.format(a, b) for a, b in
                                                   zip(X['error'], X['previous_action'])]
        X['error_previous_action_intersection'] = np.where(X['error_previous_action_intersection'].str.contains('0_'),
                                                           '0', X['error_previous_action_intersection'])
        return X['error_previous_action_intersection']
