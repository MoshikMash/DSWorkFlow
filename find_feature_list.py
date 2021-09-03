import ast


class FindFeatureList(ast.NodeVisitor):

    def __init__(self):
        self.variable_history = dict()
        self.code = None
        self.features = []

    def find_features(self, code):
        if 'feature_scaling' in code:
            print()
        self.code = code
        self.features = []
        node = ast.parse(code, mode='exec')
        self.visit(node)
        self.code = None
        if self.features:
            return self.features
        else:
            return ''

    def visit_Assign(self, node):
        # columns_to_encode = ['x', 'y']
        if isinstance(node.value, ast.List):
            self.variable_history[node.targets[0].id] = list()
            for element in node.value.elts:
                self.variable_history[node.targets[0].id].append(element.value)

        # df = FE_encode_values_of_categorical_features(df, ['x', 'y'])
        elif 'FE_encode_values' in self.code or 'FE_create_one_hot' in self.code:
            if isinstance(node.value.args[1], ast.List):
                for element in node.value.args[1].elts:
                    self.features.append(element.value)

            # df = FE_encode_values_of_categorical_features(df, columns_to_encode)
            elif isinstance(node.value.args[1], ast.Name):
                self.features = self.variable_history[node.value.args[1].id]

        # X_train_new, X_test_new = feature_scaling(X_train, X_test, ['x', 'y'], 'MinMaxScaler')
        elif 'feature_scaling' in self.code:
            if isinstance(node.value.args[2], ast.List):
                for element in node.value.args[2].elts:
                    self.features.append(element.value)
            # X_train_new, X_test_new = feature_scaling(X_train, X_test, columns_to_scale, 'MinMaxScaler')
            elif isinstance(node.value.args[2], ast.Name):
                self.features = self.variable_history[node.value.args[2].id]

        # X_train = X_train[columns]
        elif hasattr(node.value, 'slice'):
            if isinstance(node.value.slice, ast.Name):
                self.features = self.variable_history[node.value.slice.id]

        # X_train = X_train[['age', 'gender']]
        elif isinstance(node.value.slice, ast.List):
            for element in node.value.slice.elts:
                self.features.append(element.value)

