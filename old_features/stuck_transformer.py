from sklearn.base import TransformerMixin


class Stuck(TransformerMixin):
    def __init__(self):
        self.features = ['stuck']

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.features]

    def get_feature_names(self):
        return self.features

    def append_features(self, X):
        return X

    def get_feature_names(self):
        return self.features
