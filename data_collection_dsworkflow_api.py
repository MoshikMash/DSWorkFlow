from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, f1_score, accuracy_score, precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import os
import pandas as pd

pd.options.display.max_columns = None
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from IPython.display import display
import ast
from IPython.core.page import page
from IPython.core.display import HTML
import json

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


## Data loading and data splitting functions

# read the csv file and return a dataframe
def data_read_df():
    db_name = 'Adult'
    PATH = 'dataset/'
    df = pd.read_csv(os.path.join(PATH, f'{db_name}.csv'))
    return df


def data_split_train_and_test(df, test_size):
    sss = StratifiedShuffleSplit(n_splits=2, test_size=test_size, random_state=0)

    y = df['target']
    for train_index, test_index in sss.split(df, y):
        X_train, X_test = df.iloc[train_index], df.iloc[test_index]

    y_train = X_train['target']
    y_test = X_test['target']
    X_train = X_train.drop(columns=['target'])
    X_test = X_test.drop(columns=['target'])

    return X_train, X_test, y_train, y_test


# Visualiztion functions


def visual_generate_bar_char_plot(df, feature_name):
    if len(df[feature_name].unique()) > 20:
        g = sns.FacetGrid(df, aspect=4)
    else:
        g = sns.FacetGrid(df, aspect=2)
    g.map(sns.countplot, feature_name, order=list(
        df[feature_name].value_counts().index))
    g.set_xticklabels(rotation=80)
    g.set_ylabels("count")


def visual_generate_category_target_prob_plot(df, feature_name):
    if len(df[feature_name].unique()) > 20:
        x = df[feature_name]
        g = sns.catplot(x=feature_name, y="target", data=df,
                        kind="bar", palette="muted", order=list(df[feature_name].value_counts().index), aspect=4)
    else:
        g = sns.catplot(x=feature_name, y="target", data=df,
                        kind="bar", palette="muted", order=list(df[feature_name].value_counts().index), aspect=1)
    g.set_xticklabels(rotation=80)
    g.set_ylabels("Prob. for target 1")
    plt.ylim(0, 1)


def visual_generate_dis_plot(df, feature_name):
    g = sns.FacetGrid(df, col='target', aspect=1)
    g = g.map(sns.histplot, feature_name)
    g.set_xticklabels(rotation=80)


## Training Functions

def train_model(ml_algo, df_train, y_train, params=None):
    '''
    ml_algo options:
        'LogisticRegression' / 'DecisionTree' / 'KNN' / 'RandomForest' / 'MultinomialNB / GaussianNB'
    '''
    if params is None:
        params = {}
    if ml_algo == 'LogisticRegression':
        if 'multi_class' not in params:
            params['multi_class'] = 'auto'
        if 'solver' not in params:
            params['solver'] = 'liblinear'
        classifier = LogisticRegression(**params)
    elif ml_algo == 'DecisionTree':
        classifier = DecisionTreeClassifier(**params)
    elif ml_algo == 'KNN':
        classifier = KNeighborsClassifier(**params)
    elif ml_algo == 'RandomForest':
        if 'n_estimators' not in params:
            params['n_estimators'] = 100
        classifier = RandomForestClassifier(**params)
    elif ml_algo == 'MultinomialNB':
        classifier = MultinomialNB()
    elif ml_algo == 'GaussianNB':
        classifier = GaussianNB()
    else:
        raise (Exception("""The value of the ml-algo parameter should be one of the following:
                            LogisticRegression / DecisionTree / KNN / RandomForest / MultinomialNB / GaussianNB"""))

    classifier.fit(df_train, y_train)

    return classifier


# Evaluation functions

def eval_get_cm(classifier, X, y):
    cmtx = pd.DataFrame(
        confusion_matrix(classifier.predict(X), y),
        index=['pred:0', 'pred:1'],
        columns=['true:0', 'true:1']
    )
    return cmtx


def eval_get_score(classifier, X, y, metric):
    if metric == 'f1' or metric == 'auc':
        if (len(y.unique())) > 2:
            raise (Exception('You cannot use the metric \'{}\' for multiclass classification tasks'.format(metric)))
    '''
    metric options:
        'precision' / 'recall' / 'accuracy' / 'auc' / 'f1'
    '''
    if metric == 'accuracy':
        return accuracy_score(y, classifier.predict(X), normalize=True)
    elif metric == 'f1':
        return f1_score(y, classifier.predict(X))
    elif metric == 'precision':
        from sklearn.metrics import precision_score
        return precision_score(y, classifier.predict(X))
    elif metric == 'recall':
        from sklearn.metrics import recall_score
        return recall_score(y, classifier.predict(X))
    elif metric == 'auc':
        predictions_proba = classifier.predict_proba(X)[:, 1]
        fpr, tpr, t = roc_curve(y, predictions_proba)
        roc_auc = auc(fpr, tpr)
    else:
        raise (Exception("""The value of the metric parameter sholud be one of the following:
                            precision / recall / accuracy / auc / f1"""))
    return roc_auc


def eval_plot_roc_curve(classifier, X, y):
    if (len(y.unique())) > 2:
        raise (Exception('You cannot use this function for multiclass classification tasks'))
    # predict probabilities
    y_pred_prob = classifier.predict_proba(X)

    fpr, tpr, t = roc_curve(y, y_pred_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    fig1 = plt.figure(figsize=[12, 12])
    ax1 = fig1.add_subplot(111, aspect='equal')

    plt.plot(fpr, tpr, lw=2, alpha=0.3,
             label='AUC = ' + str(round(roc_auc, 2)))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")

    plt.show()


def eval_get_predictions(classifier, X, y):
    predictions = classifier.predict(X)
    predictions_df = pd.DataFrame(
        {'example_index': list(X.index), 'Pred': list(predictions), 'True_value': list(y)}).set_index('example_index')
    return predictions_df


def get_predictions_proba(classifier, X, y):
    predictions = [round(pred, 3) for pred in list(classdier.predict_proba(X)[:, 1])]
    predictions_df = pd.DataFrame(
        {'example_index': list(X.index), 'pred': predictions, 'True_value': list(y)}).set_index('example_index')
    return predictions_df


## Feature engineering (FE) functions

def FE_encode_values_of_categorical_features(df, columns_to_encode):
    df_to_return = df.copy()
    le = LabelEncoder()
    for col in columns_to_encode:
        df_to_return[col] = le.fit_transform(df_to_return[col])
    return df_to_return


def FE_create_one_hot_encodeing(df, columns_to_encode):
    for x in columns_to_encode:
        df = pd.concat([df, pd.get_dummies(df[x], prefix=x)], axis=1)

    df = df.drop(columns=columns_to_encode)
    return df


def FE_divide_numeric_feature_to_ranges(df, column_to_divide_to_ranges, number_of_ranges):
    df_to_return = df.copy()
    df_to_return[column_to_divide_to_ranges] = pd.cut(df_to_return[column_to_divide_to_ranges], number_of_ranges)

    return df_to_return


## Feature Scaling Functions

def feature_scaling(X_train, X_test, columns_to_scale, scaler_method):
    X_train_to_return = X_train.copy()
    X_test_to_return = X_test.copy()

    if scaler_method == 'Normalizer':
        scaler = Normalizer()
    elif scaler_method == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_method == 'MinMaxScaler':
        scaler = MinMaxScaler()
    else:
        raise (Exception("""The value of the scaling_method parameter sholud be one of the following:
                            Normalize / StandardScaler / MinMaxScaler"""))

    for column in columns_to_scale:
        X_train_to_return[column] = X_train_to_return[column].astype(float)
        X_test_to_return[column] = X_test_to_return[column].astype(float)

    X_train_to_return[columns_to_scale] = scaler.fit_transform(X_train_to_return[columns_to_scale])
    X_test_to_return[columns_to_scale] = scaler.transform(X_test_to_return[columns_to_scale])

    return X_train_to_return, X_test_to_return


def show_model_training_history():
    def get_le(d):
        d = ast.literal_eval(d)
        return d['lb']

    def get_oh(d):
        d = ast.literal_eval(d)
        return d['oh']

    backup_df_file = 'C:/Users/Moshik/Dropbox (BGU)/Expert vs Novices Experiment/Jupyter tracking tool/notebook examples/backup.csv'
    if not os.path.exists(backup_df_file):
        return "There are no model to present"
    backup_df = pd.read_csv(backup_df_file)
    backup_df.rename(columns={'actual_enc': 'enc', 'actual_norm': 'scale', 'all_features': 'features'}, inplace=True)
    backup_df['enc_LE'] = backup_df['enc'].apply(get_le)
    backup_df['enc_OH'] = backup_df['enc'].apply(get_oh)
    backup_df['scale'].fillna('', inplace=True)
    backup_df['binning_features'].fillna('', inplace=True)
    backup_df['hypper_params'].fillna('', inplace=True)

    backup_df = backup_df[['features', 'enc_LE', 'enc_OH', 'scale', 'ml_algo', 'binning_features', 'hypper_params']]

    dfStyler = backup_df.style.set_properties(**{'text-align': 'left'})
    dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
    return dfStyler


def train_model_with_encoding_validation(ml_algo, df_train, y_train, params=None):
    '''
    ml_algo options:
        'LogisticRegression' / 'DecisionTree' / 'KNN' / 'RandomForest' / 'LDA' / 'MultinomialNB / GaussianNB'
    '''
    if params is None:
        params = {}
    if ml_algo == 'LogisticRegression':
        if 'multi_class' not in params:
            params['multi_class'] = 'auto'
        if 'solver' not in params:
            params['solver'] = 'liblinear'
        classifier = LogisticRegression(**params)
    elif ml_algo == 'DecisionTree':
        classifier = DecisionTreeClassifier(**params)
    elif ml_algo == 'KNN':
        classifier = KNeighborsClassifier(**params)
    elif ml_algo == 'RandomForest':
        if 'n_estimators' not in params:
            params['n_estimators'] = 100
        classifier = RandomForestClassifier(**params)
    elif ml_algo == 'LDA':
        classifier = LinearDiscriminantAnalysis(**params)
    elif ml_algo == 'MultinomialNB':
        classifier = MultinomialNB()
    elif ml_algo == 'GaussianNB':
        classifier = GaussianNB()
    else:
        raise (Exception("""The value of the ml-algo parameter should be one of the following:
                            LogisticRegression / DecisionTree / KNN / RandomForest / LDA / MultinomialNB / GaussianNB"""))

    classifier.fit(df_train, y_train)

    if ml_algo in ['LogisticRegression', 'KNN', 'LDA', 'GaussianNB']:
        # get the current cell
        source_file = 'notebook_example.ipynb'
        notebook_text = open(source_file, 'r').read()
        if notebook_text == '':
            return []

        # get the training row code in the cell
        notebook = dict(json.loads(notebook_text))
        all_cells = notebook['cells']
        max_exec_count = -1
        last_cell = None
        for cell in all_cells:
            if cell['cell_type'] == 'code' and 'hide_input' not in cell['metadata']:
                if cell['outputs']:
                    for output in cell['outputs']:
                        if output['output_type'] == 'error':
                            continue
                if cell['execution_count']:
                    if cell['execution_count'] > max_exec_count:
                        max_exec_count = cell['execution_count']
                        last_cell = cell
        train_code = None
        for code in last_cell['source']:
            if 'train_model' in code:
                train_code = code

        # get le features
        if train_code:
            node = ast.parse(train_code, mode='exec')
            dfs_with_le = pd.read_csv('backup_le.csv')
            X_train_var_name = node.body[0].value.args[1].id
            if X_train_var_name in list(dfs_with_le):
                features_with_le = list(dfs_with_le[X_train_var_name].values)
                features_with_le = [f for f in features_with_le if f in list(df_train)]
                features_with_le_str = ''.join(features_with_le)
                msg = f"""You should use One-Hot Encoding instead of Label Encoding in order to get better
performance when you train a {ml_algo} model for the following features: {features_with_le_str}."""
                page(msg)

                msg_to_print = f"""You should use <b>One-Hot Encoding</b> instead of <b>Label Encoding</b> in order to
get better performance when you train a {ml_algo} model for the following features: <b>{features_with_le_str}</b>
<br/><br/> <a href="https://www.moshemash.com">Further Information</a>"""
                display(HTML(msg_to_print))
                # btn = widgets.Button(description='Medium')
                # display(btn)
                #
                # def btn_eventhandler(obj):
                #     print('Hello from the {} button!'.format(obj.description))
                #
                # btn.on_click(btn_eventhandler)

    return classifier
