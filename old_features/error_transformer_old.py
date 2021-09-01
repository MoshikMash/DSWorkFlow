from sklearn.base import TransformerMixin
import json
import pickle
from sklearn.preprocessing import OneHotEncoder


class Error(TransformerMixin):
    def __init__(self):
        self.configuration_data = json.loads(open("configuration.json", "r").read())
        self.features = ['error']
        self.enc = False
        self.error_dict = pickle.load(open("features/dicts/error_dict.p", 'rb'))

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.features]

    def get_feature_names(self):
        return self.features

    def append_features(self, X):
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
