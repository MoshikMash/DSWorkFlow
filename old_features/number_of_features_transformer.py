from sklearn.base import TransformerMixin


class NumberOfFeatures(TransformerMixin):
    def __init__(self):
        self.features = None

    def fit(self, X):
        self.features = list()
        all_the_dataframe_features = list(X.columns)
        for feature in all_the_dataframe_features:
            if feature.find('number_of_features') > -1:
                self.features.append(feature)
        return self

    def transform(self, X):
        return X[self.features]

    def get_feature_names(self):
        return self.features

    def append_features(self, X):
        return X
