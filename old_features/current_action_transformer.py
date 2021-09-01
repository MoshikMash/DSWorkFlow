from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class CurrentAction(TransformerMixin):
    def __init__(self):
        self.features = ['label_code']

        self.enc = None

    def fit(self, X):
        self.enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.enc.fit(X[self.features])
        return self

    def transform(self, X):
        return self.enc.transform(X[self.features])

    def get_feature_names(self):
        return self.features

    def append_features(self, X):
        return X

    def get_feature_names(self):
        return self.enc.get_feature_names(self.features)
