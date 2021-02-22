import pandas as pd
import pickle
import numpy as np
import re
import os


METRICS = ["accuracy", "recall", "auc", "f1", "precision", "confusion_matrix", "predictions", "accuracy_score"]

def is_number(s):
    # Return true if a string could be a float / metric
    try:
        float(s)
        return True
    except ValueError:
        return False


def getCorrTarget(data):
    print("IN CORR")
    corr_vals = []
    for index, row in data.iterrows():
        # print(index)
        label = row['label']
        code = row['code'].lower()
        execNUM = row['execTime']
        if execNUM == 6:
            print(index, row)
        outputtext = row['outputtext']
        if (label == "Visualization_corr") and (len(outputtext) > 0) and not ("<Figure" in outputtext[0]) and not ("<pandas.io.formats.style.Styler" in outputtext[0]):
            outputlist = []
            for i in range(len(outputtext)):
                line = outputtext[i]
                if ("rows" in outputtext) and ("columns" in outputtext):
                    continue
                s = line.strip().strip("\\")
                new_list = s.split()
                # print(new_list)
                if (i == 0) or ((i > 0) and (outputtext[i - 1] == "\n")):
                    L = len(s.split())
                    # print("NEW L:", L)
                if ((i > 0) and (outputtext[i - 1] == "\n")) and ("PaymentMethod_Electronic" in line):
                    ## Special case for P5T1
                    outputlist.append(['PaymentMethod_Electroniccheck'])
                    L = 1
                    # print("NEW L:", L)
                    i = i + 1
                    continue
                if ((i > 0) and (outputtext[i - 1] == "\n")) and ("MultipleLines_No" in line):
                    ## Special case for P5T1
                    outputlist.append(['MultipleLines_Nophoneservice'])
                    L = 1
                    # print("NEW L:", L)
                    i = i + 1
                    continue

                if ((i > 0) and (outputtext[i - 1] == "\n")) and ("PaymentMethod_Mailed" in line):
                    ## Special case for P5T1
                    outputlist.append(['PaymentMethod_Mailedcheck'])
                    L = 1
                    # print("NEW L:", L)
                    i = i + 1
                    continue
                if ((i > 0) and (outputtext[i - 1] == "\n")) and ("PaymentMethod_Bank" in line):
                    ## Special case for P5T1
                    outputlist.append(['PaymentMethod_Banktransfer(automatic)'])
                    L = 1
                    # print("NEW L:", L)
                    i = i + 2
                    continue
                if ((i > 0) and (outputtext[i - 1] == "\n")) and ("PaymentMethod_Credit" in line):
                    ## Special case for P5T1
                    outputlist.append(['PaymentMethod_Creditcard(automatic)'])
                    L = 1
                    # print("NEW L:", L)
                    i = i + 2
                    continue
                if ((i > 0) and (outputtext[i - 1] == "\n")) and ("Contract_One" in line):
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
                outputlist = outputlist[index+1:]
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
                df = pd.DataFrame.from_records(item, columns = column_names)
                full_df = pd.concat([full_df, df], axis=1)
                # print(full_df)
            full_df = full_df.set_index('row_name')
            # print("BEFORE DROP:", full_df.columns)
            try: full_df = full_df.drop(['row_name'], axis=1)
            except: pass
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
                    try: max_ind = max_ind[0]
                    except: pass
                    # print(max_ind)
                    dic[col_name] = (max_ind, max_val)
            corr_vals.append(dic)
            if bool(dic):
                print(index, execNUM, dic)
                print(corr_vals)
        else:
            corr_vals.append(0)
    return corr_vals


def addFeatures(features):
    for i in range(len(features)):
        val = features[i].strip().strip("'").strip("\"")
        features[i] = val
    return features

def subFeatures(features):
    for i in range(len(features)):
        val = features[i].strip().strip("'").strip("\"")
        features[i] = "-" + val
    return features

def twoLineFeatures(features, data, index, var_name, new_label_d, feature_d):
    # print("INDEX", index)
    next_line = data.iloc[index + 1]['code'].lower()
    # print("NEXT LINE:", next_line)
    next_label = data.iloc[index + 1]['label']
    if (next_label == "FE_binning") or (next_label == "FE_encoding") or ("feature_scaling" in next_label):
        # this is not a feature selection list
        return new_label_d, feature_d
    if (var_name in next_line) and (".drop(" in next_line):
        # We  have a subtractive two line selection
        features = subFeatures(features)
        new_label_d[index + 1] = "feature_sel"
        feature_d[index + 1] = features
        return new_label_d, feature_d
    elif (var_name in next_line) and not (".drop(" in next_line):
        # We  have an additive two line selection
        features = addFeatures(features)
        new_label_d[index + 1] = "feature_sel"
        feature_d[index + 1] = features
        return new_label_d, feature_d
    else: return new_label_d, feature_d

def findFinishedLine(list_s, data, index):
    code = list_s
    list_ls = [list_s[0]]
    i = index + 1
    while not("]]" in code) and not("]" in code):
        code = data.iloc[i, 0].lower()
        list_ls.append(code)
        i = i+1
    list_sub = ""
    for ls in list_ls:
        list_sub = list_sub + ls.strip("\[").strip("\]")
    return list_sub, i - 1



def findFeatures(data):
    feature_d = dict()
    new_label_d = dict()
    not_prev_line = data[(data.label == "FE_encoding") | (data.label == "FE_binning") | (data.label == "feature_scaling")].index
    possible_indices = data[(data.label == "OTHER") | (data.label == "feature_sel")].index
    not_lines = not_prev_line - 1
    possible_indices = possible_indices.difference(not_lines)
    sub = data.iloc[possible_indices]
    for index, row in sub.iterrows():
        code = row['code'].lower()
        code = code.strip()
        if code == "":
            continue
        if ("=" in code) and not(".drop" in code):
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
            list_sub = code[code.find("["):code.find("]")+1]
        else:
            continue
        list = re.findall("\[[\"'].*", list_sub)
        if ("[[" in list_sub) and (len(list) == 1):
            # Found in place, additive selection
            if "]]" not in list_sub:
                # look at next lines
                list, i = findFinishedLine(list, data, index)
                list = [list]
            features = re.findall("[\"'][^\"']*[\"']", list[0])
            features = addFeatures(features)
            new_label_d[index] = "feature_sel"
            feature_d[index] = features
            new_label_d[index] = "feature_sel"
        elif (".drop(" in code) and (len(list) == 1):
            # found in place subtractive selection
            features = re.findall("[\"'][^\"']*[\"']", list[0])
            # print(features)
            features = subFeatures(features)
            new_label_d[index] = "feature_sel"
            feature_d[index] = features
        elif len(list) == 1:
            # We may have have a two line selection
            if "]" not in list_sub:
                # look at next lines
                list, index = findFinishedLine(list, data, index)
                list = [list]
            features = re.findall("[\"'][^\"']*[\"']", list[0])
            if index <= data.shape[0]:
                new_label_d, feature_d = twoLineFeatures(features, data, index, var_name, new_label_d, feature_d)
    features_col = []
    for index, row in data.iterrows():
        # Find score or type in dictionary, or return default -1
        features_col.append(feature_d.get(index, -1))
    for key in new_label_d:
        # Find score or type in dictionary, or return default -1
        data.loc[key, "label"] = new_label_d[key]
    return features_col





def getMoreMetricD(data):
    scores_d = dict()
    data_types_d = dict()
    e_sub = data[(data.label == "eval") & (data.has_error != 1)]
    for index, row in e_sub.iterrows():
        metric = 0
        code = row['code'].lower()
        function_call = code[code.find("eval_get_score(")+1:code.find(")")]
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

def getMoreMetric(data):
    data_types_d, scores_d = getMoreMetricD(data)
    data_types = []
    scores = []
    for index, row in data.iterrows():
        # Find score or type in dictionary, or return default -1
        scores.append(scores_d.get(index, -1))
        data_types.append(data_types_d.get(index, -1))
    return data_types, scores


def getLabel(value, object, i):
    label = "OTHER"
    if ("import" in value):
        # print("GOT ONE")
        label = "import"
    if ("visual" in value):
        # print("GOT ONE")
        label = "visualize"
    if (("head" in value) or ("print" in value)):
        label = "print"
    if ((".describe" in value) or (".size" in value) or (".value_counts" in value) or
            (".dtypes" in value) or
            (".count" in value)):
        label = "describe_dataframe"
    if (("FE" in value) and ("encode" in value)):
        # print("GOT ENCODE: ", value)
        label = "FE_encoding"
    if (("FE" in value) and ("range" in value) or ("np.where" in value)):
        # print("GOT BINNING: ", value)
        label = "FE_binning"
    if ("feature_scaling" in value):
        if ("MinMaxScaler" in value):
            type = "MinMaxScaler"
        elif ("StandardScaler" in value):
            type = "StandardScaler"
        else:
            type = "Normalizer"
        label = "feature_scaling" + type
    if ("[[" in value):
        label = "feature_sel"
    if ("train_model" in value):
        label = "train / hyperparameter_tuning"
    if (("eval_" in value) and not ("plot" in value)):
        label = "eval"
    if ((".coef" in value) or (("eval" in value) and ("plot" in value))):
        label = "evaluation_visualization"
    if ("split" in value):
        label = "splitting"
    if (".corr" in value):
        label = "Visualization_corr"
    if (value == ""):
        label = "empty"
    return label


def add_to_dict(sub, numerics, metric_d):
    i = 0
    for index, row in sub.iterrows():
        # Build metric dictionary based on index (withoriginal dataframe)
        metric_d[index] = numerics[i]
        i = i + 1
    return metric_d


def getMetricD(data):
    # Build Metric Dictionary mapping line index (in data frame) to metric value
    metric_d = {}
    for execTime in data.execTime.unique():
        sub = data[data.execTime == execTime]
        text = sub.outputtext.tolist()[0]
        numerics = []
        for i in range(0, len(text)):
            # Break up text if metrics (decimals) on multiple lines
            text_str = text[i]
            if "axes" in text_str.lower():
                # This is a plot, with axes
                break
            numerics.extend([float(s) for s in text_str.split() if is_number(s)])
        if len(numerics) > 0:
            e_sub = sub[sub.label == "eval"]
            no_e_sub = sub[sub.label != "eval"]
            if e_sub.shape[0] == len(numerics):
                # All evaluated metrics are printed
                metric_d = add_to_dict(e_sub, numerics, metric_d)
            elif e_sub.shape[0] > len(numerics):
                # An evaluated metric is not printed
                used = []
                for index, row in e_sub.iterrows():
                    # Find the metrics both in the evaluations and the output text
                    code = row['code'].lower()
                    text = row['outputtext'][0].lower()
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
                new_sub = data[data.execTime == execTime - 1]
                new_sub = new_sub[(new_sub.label == "eval") & (new_sub.score != "confusion_matrix") & (new_sub.score != "predictions") & (new_sub.has_error == 0)]
                if new_sub.shape[0] == len(numerics):
                    metric_d = add_to_dict(new_sub, numerics, metric_d)
    # print("METRIC DICT:", metric_d)
    return metric_d

def getMetric(data):
    metric_D = getMetricD(data)
    metrics = []
    for index, row in data.iterrows():
        # Find metric in metric dictionary, or return default -1
        metrics.append(metric_D.get(index,-1))
    # print(metrics)
    return metrics

def getCMContent(data):
    metric_d = {}
    cm_sub = data[data.score == "confusion_matrix"]
    print(cm_sub[["code", "outputtext"]])
    for execTime in cm_sub.execTime.unique():
        print(execTime)
        sub = cm_sub[cm_sub.execTime == execTime]
        sub_all = data[(data.execTime == execTime) & (data.score != "confusion_matrix")]
        used_nums = sub_all["metric"].tolist()
        text = sub.outputtext.tolist()[0]
        numerics = []
        for i in range(0, len(text)):
            # Break up text if multiple print outs
            text_str = text[i]
            if "axes" in text_str.lower():
                # This is a plot, with axes
                break
            numerics.extend([float(s) for s in text_str.split() if is_number(s)])
        numerics = list(set(numerics).difference(set(used_nums)))
        e_sub = sub[(sub.label == "eval")]
        no_e_sub = sub[sub.label != "eval"]
        if (len(numerics) == 4) and (e_sub.shape[0] == 1):
            # Confusion matrix is printed in same code chunk as its created
            metric_d = add_to_dict(e_sub, [numerics], metric_d)
            print(metric_d)
        elif (e_sub.shape[0] >= 1) and  (len(numerics) <= 4):
            new_sub = data[data.execTime == execTime + 1]
            text = new_sub.outputtext.tolist()[0]
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
                    metric_d = add_to_dict(sub.iloc[[i]], [numerics[(0 + (i*4)):(4 + (i*4))]], metric_d)
    cm = []
    for index, row in data.iterrows():
        # Find metric in metric dictionary, or return default -1
        cm.append(metric_d.get(index, -1))
    # print(metrics)
    return cm

def getModelType(data):
    sub =  data[(data.label == "train") | (data.label == "hyperparameter_tuning")]
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
    print(model_types)
    return model_types




def mapExecutions(filepath, outfile):
    object = pickle.load(open(filepath, 'rb'))
    labels = ["OTHER"] * len(object)
    get_cm = [0] * len(object)
    errors = [0] * len(object)
    metrics = [0] * len(object)
    for i in range(len(object)):
        value = object[i][0]
        labels[i] = getLabel(value, object, i)
    data = pd.DataFrame(data=object, columns=["code", "execTime", "cellNum", "filename", "outputtype","outputtext", "has_error", "error_type"]) #"code", "execTime", "filename"])
    data['label'] = labels
    data_types, scores, = getMoreMetric(data)
    data['score'] = scores
    data['metric_type'] = data_types
    data['metric'] = getMetric(data)
    data['cm_content'] = getCMContent(data)
    data['features'] = findFeatures(data)
    data = data[['code', 'execTime', 'cellNum', 'filename', "metric", "has_error", "label", "error_type", "score", "metric_type", "cm_content", "features"]] #, "cell_number", "hpt_cell_number"]]
    print(data[['code', 'has_error']])
    with open(outfile, 'wb') as f:
       pickle.dump(data, f)


mapExecutions("parsed/P1T3.p", outfile = 'labeled/P1_task3_nohand_labeled.pickle')
