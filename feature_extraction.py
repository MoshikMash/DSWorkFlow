import pandas as pd
import re
from copy import deepcopy
import workflow_reconstruction
import json

from features.train_eval_loop_features.required_information import train_eval_loop_transformer
from features.train_eval_loop_features import counter_train_eval_loops_transformer, \
    counter_train_eval_loops_with_time_window_transformer

from features.action_features.required_information import action_changed_between_train_eval_loops_transformer
from features.action_features import counters_with_time_window_size_action_changed_between_train_eval_loops_transformer, \
    action_counter_with_time_window_transformer

from features.performance_features.required_information import best_score_and_improvement_transformer
from features.performance_features import performance_improved_with_time_window_size_transformer

from features.inactivity_features.required_information import inactivity_moments_transformer
from features.inactivity_features import inactivity_counter_with_time_window_transformer, \
    inactivity_maximum_with_time_window_transformer

from features.integrated_features import action_counters_AND_train_eval_loops, \
    counter_at_least_one_of_metrics_was_improved_with_time_window_size_transformer

from features.irrational_behavior.required_information import train_repetition_transformer
from features.irrational_behavior import train_repetition_counter_with_time_window_transformer

METRICS = ["accuracy", "recall", "auc", "f1", "precision", "confusion_matrix", "predictions", "accuracy_score"]
configuration_data = json.loads(open("configuration.json", "r").read())


def is_number(s):
    # Return true if a string could be a float / metric
    try:
        float(s)
        return True
    except ValueError:
        return False


def add_to_dict(sub, numerics, metric_d):
    i = 0
    for index, row in sub.iterrows():
        # Build metric dictionary based on index (withoriginal dataframe)
        metric_d[index] = numerics[i]
        i = i + 1
    return metric_d


def get_more_metric_d(data):
    scores_d = dict()
    data_types_d = dict()
    e_sub = data[(data.action == "eval") & (data.has_error != 1)]
    for index, row in e_sub.iterrows():
        metric = 0
        code = row['code'].lower()
        function_call = code[code.find("eval_get_score(") + 1:code.find(")")]
        params = function_call.split(",")
        X_name = params[1]
        y_name = params[2]
        if "get_cm" in function_call:
            metric = "confusion_matrix"
        elif "get_predictions" in function_call:
            metric = "predictions"
        else:
            metric = params[3].strip()
            metric = metric[1:-1]
        if metric not in METRICS:
            print("ERROR Index:", index)
            print(metric)
            print("has_error:", row['has_error'])
        scores_d[index] = metric
        if "train" in X_name and "train" in y_name:
            data_type = "train"
        elif "test" in X_name and "test" in y_name:
            data_type = "test"
        else:
            print("X", X_name)
            print("Y", y_name)
            data_type = "?????"
        data_types_d[index] = data_type
    return data_types_d, scores_d


def find_finished_line(list_s, data, index):
    code = list_s
    list_ls = [list_s[0]]
    i = index + 1
    while not ("]]" in code) and not ("]" in code):
        code = data.iloc[i, 0].lower()
        list_ls.append(code)
        i = i + 1
    list_sub = ""
    for ls in list_ls:
        list_sub = list_sub + ls.strip("\[").strip("\]")
    return list_sub, i - 1


def two_line_features(features, data, index, var_name, new_action_d, feature_d):
    # print("INDEX", index)
    next_line = data.loc[index + 1]['code'].lower()
    # print("NEXT LINE:", next_line)
    next_action = data.loc[index + 1]['action']
    if (next_action == "FE_binning") or (next_action == "FE_encoding") or ("feature_scaling" in next_action):
        # this is not a feature selection list
        return new_action_d, feature_d
    if (var_name in next_line) and (".drop(" in next_line):
        # We  have a subtractive two line selection
        features = sub_features(features)
        new_action_d[index + 1] = "feature_sel"
        feature_d[index + 1] = features
        return new_action_d, feature_d
    elif (var_name in next_line) and not (".drop(" in next_line):
        # We  have an additive two line selection
        features = add_features(features)
        new_action_d[index + 1] = "feature_sel"
        feature_d[index + 1] = features
        return new_action_d, feature_d
    else:
        return new_action_d, feature_d


def add_features(features):
    for i in range(len(features)):
        val = features[i].strip().strip("'").strip("\"")
        features[i] = val
    return features


def sub_features(features):
    for i in range(len(features)):
        val = features[i].strip().strip("'").strip("\"")
        features[i] = "-" + val
    return features


def get_score_d(data):
    # Build Metric Dictionary mapping line index (in data frame) to metric value
    metric_d = {}
    for exec_count in data.exec_count.unique():
        sub = data[data.exec_count == exec_count]
        text = sub.output_text.tolist()[0]
        numerics = []
        for i in range(0, len(text)):
            # Break up text if metrics (decimals) on multiple lines
            text_str = text[i]
            if "axes" in text_str.lower():
                # This is a plot, with axes
                break
            numerics.extend([float(s) for s in text_str.split() if is_number(s)])
        if len(numerics) > 0:
            e_sub = sub[sub.action == "eval"]
            if e_sub.shape[0] == len(numerics):
                # All evaluated metrics are printed
                metric_d = add_to_dict(e_sub, numerics, metric_d)
            elif e_sub.shape[0] > len(numerics):
                # An evaluated metric is not printed
                used = []
                for index, row in e_sub.iterrows():
                    # Find the metrics both in the evaluations and the output text
                    code = row['code'].lower()
                    text = row['output_text'][0].lower()
                    for metric in METRICS:
                        if (metric in code) and (metric in text):
                            used.append(metric)
                new_df = pd.DataFrame()
                for index, row in e_sub.iterrows():
                    # For the used metrics, take only the relevant rows
                    m_flag = 0
                    code = row['code'].lower()
                    for metric in used:
                        if (metric in code):
                            m_flag = 1
                    if m_flag:
                        new_df = new_df.append(row)
                metric_d = add_to_dict(new_df, numerics, metric_d)
            elif (e_sub.shape[0] < len(numerics)):
                new_sub = data[data.exec_count == exec_count - 1]
                new_sub = new_sub[(new_sub.action == "eval") & (new_sub.metric != "confusion_matrix") & (
                        new_sub.metric != "predictions") & (new_sub.has_error == 0)]
                if new_sub.shape[0] == len(numerics):
                    metric_d = add_to_dict(new_sub, numerics, metric_d)
    # print("METRIC DICT:", metric_d)
    return metric_d


class FeatureExtraction:
    def __init__(self, reconstructed_df):
        self.df = deepcopy(reconstructed_df.df)
        self.df = self.df[self.df['exec_count'] > 1]

    # static methods
    @staticmethod
    def find_features(data):
        feature_d = dict()
        new_action_d = dict()
        not_prev_line = data[
            (data.action == "FE_encoding") | (data.action == "FE_binning") | (data.action == "feature_scaling")].index
        possible_indices = data[(data.action == "OTHER") | (data.action == "feature_sel")].index
        not_lines = not_prev_line - 1
        possible_indices = possible_indices.difference(not_lines)
        sub = data.loc[possible_indices]
        for index, row in sub.iterrows():
            code = row['code'].lower()
            code = code.strip()
            if code == "":
                continue
            if ("=" in code) and not (".drop" in code):
                try:
                    var_name = code.split("=")[0].strip()
                    list_sub = code.split("=")[1]
                except:
                    var_name = code.split("=")[0].strip()
                    list_sub = ""
                if "[" in var_name:
                    # print("BROKEN", var_name, code)
                    continue
            elif (".drop" in code):
                # print("USED IT")
                list_sub = code[code.find("["):code.find("]") + 1]
            else:
                continue
            list = re.findall("\[[\"'].*", list_sub)
            if ("[[" in list_sub) and (len(list) == 1):
                # Found in place, additive selection
                if "]]" not in list_sub:
                    # look at next lines
                    list, i = find_finished_line(list, data, index)
                    list = [list]
                features = re.findall("[\"'][^\"']*[\"']", list[0])
                features = add_features(features)
                new_action_d[index] = "feature_sel"
                feature_d[index] = features
                new_action_d[index] = "feature_sel"
            elif (".drop(" in code) and (len(list) == 1):
                # found in place subtractive selection
                features = re.findall("[\"'][^\"']*[\"']", list[0])
                # print(features)
                features = sub_features(features)
                new_action_d[index] = "feature_sel"
                feature_d[index] = features
            elif len(list) == 1:
                # We may have have a two line selection
                if "]" not in list_sub:
                    # look at next lines
                    list, index = find_finished_line(list, data, index)
                    list = [list]
                features = re.findall("[\"'][^\"']*[\"']", list[0])
                if index <= data.shape[0]:
                    new_action_d, feature_d = two_line_features(features, data, index, var_name, new_action_d,
                                                                feature_d)
        features_col = []
        for index, row in data.iterrows():
            # Find score or type in dictionary, or return default -1
            features_col.append(feature_d.get(index, -1))
        for key in new_action_d:
            # Find score or type in dictionary, or return default -1
            data.loc[key, "action"] = new_action_d[key]
        return features_col

    @staticmethod
    def get_more_metric(data):
        data_types_d, scores_d = get_more_metric_d(data)
        data_types = []
        scores = []
        for index, row in data.iterrows():
            # Find score or type in dictionary, or return default -1
            scores.append(scores_d.get(index, -1))
            data_types.append(data_types_d.get(index, -1))
        return data_types, scores

    @staticmethod
    def get_action(code):
        action = "OTHER"
        if "import" in code:
            # print("GOT ONE")
            action = "import"
        if "visual" in code:
            # print("GOT ONE")
            action = "visualize"
        if ("head" in code) or ("print" in code):
            action = "print"
        if ((".describe" in code) or (".size" in code) or (".value_counts" in code) or
                (".dtypes" in code) or
                (".count" in code)):
            action = "describe_dataframe"
        if ("FE" in code) and ("encode" in code):
            # print("GOT ENCODE: ", code)
            action = "FE_encoding"
        if ("FE" in code) and ("range" in code) or ("np.where" in code):
            # print("GOT BINNING: ", code)
            action = "FE_binning"
        if "feature_scaling" in code:
            if "MinMaxScaler" in code:
                scaling_type = "MinMaxScaler"
            elif "StandardScaler" in code:
                scaling_type = "StandardScaler"
            else:
                scaling_type = "Normalizer"
            action = "feature_scaling" + '_' + scaling_type
        if "[[" in code:
            action = "feature_sel"
        if "train_model(" in code:
            action = "train / hyperparameter_tuning"
        if ("eval_" in code) and not ("plot" in code):
            action = "eval"
        if (".coef" in code) or (("eval" in code) and ("plot" in code)):
            action = "evaluation_visualization"
        if "split" in code:
            action = "splitting"
        if ".corr" in code:
            action = "Visualization_corr"
        if code == "":
            action = "empty"
        if 'def' in code:
            action = 'OTHER'
        return action

    @staticmethod
    def get_score(data):
        socre_D = get_score_d(data)
        scores = []
        for index, row in data.iterrows():
            # Find metric in metric dictionary, or return default -1
            scores.append(socre_D.get(index, -1))
        # print(metrics)
        return scores

    @staticmethod
    def get_cm_content(data):
        metric_d = {}
        cm_sub = data[data.score == "confusion_matrix"]
        for exec_count in cm_sub.exec_count.unique():
            sub = cm_sub[cm_sub.exec_count == exec_count]
            sub_all = data[(data.exec_count == exec_count) & (data.score != "confusion_matrix")]
            used_nums = sub_all["metric"].tolist()
            text = sub.output_text.tolist()[0]
            numerics = []
            for i in range(0, len(text)):
                # Break up text if multiple print outs
                text_str = text[i]
                if "axes" in text_str.lower():
                    # This is a plot, with axes
                    break
                numerics.extend([float(s) for s in text_str.split() if is_number(s)])
            numerics = list(set(numerics).difference(set(used_nums)))
            e_sub = sub[(sub.action == "eval")]
            no_e_sub = sub[sub.action != "eval"]
            if (len(numerics) == 4) and (e_sub.shape[0] == 1):
                # Confusion matrix is printed in same code chunk as its created
                metric_d = add_to_dict(e_sub, [numerics], metric_d)
            elif (e_sub.shape[0] >= 1) and (len(numerics) <= 4):
                new_sub = data[data.exec_count == exec_count + 1]
                text = new_sub.output_text.tolist()[0]
                numerics = []
                for i in range(0, len(text)):
                    # Break up text if multiple print outs
                    text_str = text[i]
                    if "axes" in text_str.lower():
                        # This is a plot, with axes
                        break
                    numerics.extend([float(s) for s in text_str.split() if is_number(s)])
                if len(numerics) / 4 == sub.shape[0]:
                    for i in range(sub.shape[0]):
                        metric_d = add_to_dict(sub.iloc[[i]], [numerics[(0 + (i * 4)):(4 + (i * 4))]], metric_d)
        cm = []
        for index, row in data.iterrows():
            # Find metric in metric dictionary, or return default -1
            cm.append(metric_d.get(index, -1))
        # print(metrics)
        return cm

    @staticmethod
    def get_model_type(data):
        sub = data[(data.action == "train") | (data.action == "hyperparameter_tuning")]
        model_d = {}
        for index, row in sub.iterrows():
            code = row['code']
            function_call = code[code.find("train_model(") + 1:code.find(")")]
            params = function_call.split("(")[1]
            val = params.split(",")[0]
            val = val[1:-1]
            model_d[index] = val
        model_types = []
        for index, row in data.iterrows():
            # Find metric in metric dictionary, or return default -1
            model_types.append(model_d.get(index, -1))
        return model_types

    @staticmethod
    def get_corr_target(data):
        print("IN CORR")
        corr_vals = []
        for index, row in data.iterrows():
            # print(index)
            action = row['action']
            code = row['code'].lower()
            execNUM = row['exec_count']
            if execNUM == 6:
                print(index, row)
            output_text = row['output_text']
            if (action == "Visualization_corr") and (len(output_text) > 0) and not (
                    "<Figure" in output_text[0]) and not (
                    "<pandas.io.formats.style.Styler" in output_text[0]):
                outputlist = []
                for i in range(len(output_text)):
                    line = output_text[i]
                    if ("rows" in output_text) and ("columns" in output_text):
                        continue
                    s = line.strip().strip("\\")
                    new_list = s.split()
                    # print(new_list)
                    if (i == 0) or ((i > 0) and (output_text[i - 1] == "\n")):
                        L = len(s.split())
                        # print("NEW L:", L)
                    if ((i > 0) and (output_text[i - 1] == "\n")) and ("PaymentMethod_Electronic" in line):
                        ## Special case for P5T1
                        outputlist.append(['PaymentMethod_Electroniccheck'])
                        L = 1
                        # print("NEW L:", L)
                        i = i + 1
                        continue
                    if ((i > 0) and (output_text[i - 1] == "\n")) and ("MultipleLines_No" in line):
                        ## Special case for P5T1
                        outputlist.append(['MultipleLines_Nophoneservice'])
                        L = 1
                        # print("NEW L:", L)
                        i = i + 1
                        continue

                    if ((i > 0) and (output_text[i - 1] == "\n")) and ("PaymentMethod_Mailed" in line):
                        ## Special case for P5T1
                        outputlist.append(['PaymentMethod_Mailedcheck'])
                        L = 1
                        # print("NEW L:", L)
                        i = i + 1
                        continue
                    if ((i > 0) and (output_text[i - 1] == "\n")) and ("PaymentMethod_Bank" in line):
                        ## Special case for P5T1
                        outputlist.append(['PaymentMethod_Banktransfer(automatic)'])
                        L = 1
                        # print("NEW L:", L)
                        i = i + 2
                        continue
                    if ((i > 0) and (output_text[i - 1] == "\n")) and ("PaymentMethod_Credit" in line):
                        ## Special case for P5T1
                        outputlist.append(['PaymentMethod_Creditcard(automatic)'])
                        L = 1
                        # print("NEW L:", L)
                        i = i + 2
                        continue
                    if ((i > 0) and (output_text[i - 1] == "\n")) and ("Contract_One" in line):
                        ## Special case for P5T1
                        outputlist.append(['Contract_Oneyear', 'Contract_Twoyear'])
                        L = 1
                        # print("NEW L:", L)
                        i = i + 3
                        continue
                    if len(new_list) > L + 1:
                        # print(new_list)
                        name = ""
                        newer_list = []
                        for elem in new_list:
                            if not is_number(elem):
                                name = name + elem
                                # print(name)
                            else:
                                # print("ELEML", elem)
                                # print("NEWER LIST:", newer_list)
                                newer_list.append(elem)
                        newer_list.insert(0, name)
                        # print(newer_list)
                        outputlist.append(newer_list)
                    else:
                        # print(new_list)
                        outputlist.append(new_list)
                # print("FINAL:", outputlist)
                biglist = []
                while [] in outputlist:
                    # print(outputlist)
                    # print([] in outputlist)
                    index = outputlist.index([])
                    biglist.append(outputlist[0:index])
                    # print(outputlist[0:index])
                    outputlist = outputlist[index + 1:]
                # print("out of while loop")
                biglist.append(outputlist)
                # print(biglist)
                # L = len(biglist)
                full_df = pd.DataFrame()
                # print(full_df)
                for item in biglist:
                    # print("ITEM:", item)
                    column_names = item.pop(0)
                    # print("COL NAMES:", column_names)
                    column_names.insert(0, "row_name")
                    df = pd.DataFrame.from_records(item, columns=column_names)
                    full_df = pd.concat([full_df, df], axis=1)
                    # print(full_df)
                full_df = full_df.set_index('row_name')
                # print("BEFORE DROP:", full_df.columns)
                try:
                    full_df = full_df.drop(['row_name'], axis=1)
                except:
                    pass
                # print("AFGTER DROP:", full_df.columns)
                full_df.to_csv('test_corr.csv', sep='\t')
                full_df = full_df.replace("...", -10)
                full_df = full_df.apply(pd.to_numeric)
                full_df = full_df.replace(1.000, -10)
                # print(full_df)
                dic = dict()
                # print(full_df.columns.tolist())
                for col_name in full_df.columns:
                    if col_name != "row_name":
                        max_val = full_df[col_name].max()
                        max_ind = full_df[col_name].idxmax()
                        try:
                            max_ind = max_ind[0]
                        except:
                            pass
                        # print(max_ind)
                        dic[col_name] = (max_ind, max_val)
                corr_vals.append(dic)
                if bool(dic):
                    print(index, execNUM, dic)
                    print(corr_vals)
            else:
                corr_vals.append(0)
        return corr_vals

    @staticmethod
    def group_labels(df):
        action_groups = configuration_data['action_groups']
        actions = df['action'].values
        updated_labels = list()
        for action in actions:
            value_changed_flag = False
            for group_name in action_groups:
                if action in action_groups[group_name]:
                    updated_labels.append(group_name)
                    value_changed_flag = True
            if value_changed_flag is False:
                updated_labels.append(action)

        return updated_labels

    # regular methods
    def extract_features(self):
        self.df['action'] = self.df['code'].apply(self.get_action)
        data_types, metrics, = self.get_more_metric(self.df)
        self.df['specific_action'] = deepcopy(self.df['action'])
        self.df['action'] = self.group_labels(self.df)

        self.df['metric'] = metrics
        self.df['metric_type'] = data_types
        self.df['score'] = self.get_score(self.df)
        self.df['cm_content'] = self.get_cm_content(self.df)
        self.df['features'] = self.find_features(self.df)

    def extract_advanced_features(self):
        feature_transformers_required_information = [
            train_eval_loop_transformer.TrainEvalLoop(),
            action_changed_between_train_eval_loops_transformer.ActionChangedBetweenTrainEvalLoops(),
            best_score_and_improvement_transformer.BestScoreAndImprovement(),
            inactivity_moments_transformer.InactivityMoments(),
            train_repetition_transformer.TrainRepetition()
        ]

        feature_transformers = []
        if configuration_data['feature_vector']['train_eval_loop_features']:
            feature_transformers.append(counter_train_eval_loops_transformer.CounterTrainEvalLoops())
            feature_transformers.append(
                counter_train_eval_loops_with_time_window_transformer.CounterTrainEvalLoopsWithTimeWindow())
        if configuration_data['feature_vector']['action_features']:
            feature_transformers.append(
                counters_with_time_window_size_action_changed_between_train_eval_loops_transformer.CountersWithTimeWindowSizeActionChangedBetweenTrainEvalLoops())
            feature_transformers.append(action_counter_with_time_window_transformer.ActionCounterWithWindow())
        if configuration_data['feature_vector']['performance_features']:
            feature_transformers.append(
                performance_improved_with_time_window_size_transformer.PerformanceImprovementWithTimeWindowSize())
        if configuration_data['feature_vector']['inactivity_features']:
            feature_transformers.append(inactivity_counter_with_time_window_transformer.InactivityCounterInWindow())
            feature_transformers.append(inactivity_maximum_with_time_window_transformer.InactivityMaxWithWindow())
        if configuration_data['feature_vector']['irrational_behavior_features']:
            feature_transformers.append(
                train_repetition_counter_with_time_window_transformer.TrainRepetitionCounterWithWindow())

        integrated_feature_transformers = []
        if configuration_data['feature_vector']['integrated_features']:
            integrated_feature_transformers.append(
                action_counters_AND_train_eval_loops.IntegratedActionCounterANDTrainEvalLoopCounter()),
            integrated_feature_transformers.append(
                counter_at_least_one_of_metrics_was_improved_with_time_window_size_transformer.CounterAtLeastOneImprovedScore()),

        all_transformers = feature_transformers_required_information + feature_transformers + integrated_feature_transformers

        for transformer in all_transformers:
            transformer.append_features(self.df)
