import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import json

# Extra Tree classifier
def extra_tree(patient_num):
    if patient_num == 1:
        return ExtraTreesClassifier(
            n_estimators=3000, random_state=0, max_depth=11, n_jobs=2
        )
    elif patient_num == 2:
        return ExtraTreesClassifier(
            n_estimators=5000,
            random_state=0,
            max_depth=15,
            n_jobs=2,
            criterion="entropy",
        )
    elif patient_num == 3:
        return ExtraTreesClassifier(
            n_estimators=4500,
            random_state=0,
            max_depth=15,
            criterion="entropy",
            n_jobs=2,
        )


# Random Forest
def random_forest(patient_num):
    if patient_num == 1:
        return RandomForestClassifier(
            n_estimators=5000,
            random_state=0,
            max_depth=15,
            n_jobs=2,
            criterion="gini",
            min_samples_split=7,
        )
    elif patient_num == 2:
        return RandomForestClassifier(
            n_estimators=5000,
            random_state=0,
            max_depth=15,
            n_jobs=2,
            criterion="gini",
            min_samples_split=7,
        )
    elif patient_num == 3:
        return RandomForestClassifier(
            n_estimators=4500,
            random_state=0,
            max_depth=15,
            criterion="gini",
            n_jobs=2,
            min_samples_split=7,
        )


# Merge feature set of an array of seleted channel
def merge_feature_set(channelSet=[]):
    feat = json.load(open("SETTINGS.json"))
    featurepath = feat["feature_path"]
    pat = feat["pat"]
    study_mode = feat["study_mode"]
    train_feature_set = ""
    test_feature_set = ""

    for i in range(len(channelSet)):
        for mode in ["train", "test"]:
            filename = "{}_pat_{}_study_{}_channel_{}.csv".format(
                mode, pat, study_mode, channelSet[i]
            )

            path = featurepath + filename
            channel_data = pd.read_csv(path, header=0)

            if i > 0:
                channel_data = channel_data.drop(["label"], axis=1)
                if mode == "train":
                    train_feature_set = pd.merge(
                        train_feature_set, channel_data, on="File"
                    )
                else:
                    test_feature_set = pd.merge(
                        test_feature_set, channel_data, on="File"
                    )
            else:
                if mode == "train":
                    train_feature_set = channel_data
                else:
                    test_feature_set = channel_data

    return train_feature_set, test_feature_set


# function to train model and generate AUC:
def predict_seizure(train_data, test_data):

    train = train_data
    test = test_data

    # clean the training data by removing nans
    train.dropna(thresh=train.shape[1] - 3, inplace=True)

    train.replace([np.inf, -np.inf], np.nan, inplace=True)
    test.replace([np.inf, -np.inf], np.nan, inplace=True)

    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)

    # get labels
    label_train = train["label"].values
    label_test = test["label"].values

    train_feat = train.drop(["File", "label"], axis=1)
    test_feat = test.drop(["File", "label"], axis=1)
    train_feat = train_feat.values
    test_feat = test_feat.values

    feat = json.load(open("SETTINGS.json"))
    # generate model for classification
    if feat["classifier"] == "ExtraTrees":
        model = extra_tree(patient_num)
    elif feat["classifier"] == "LogisticRegression":
        model = LogisticRegression(max_iter=2000)
    elif feat["classifier"] == "RandomForest":
        model = random_forest(patient_num)
    elif feat["classifier"] == "LDA":
        model = LinearDiscriminantAnalysis()

    model.fit(train_feat, label_train)
    y_pred = model.predict_proba(test_feat)

    # return AUC value
    return metrics.roc_auc_score(label_test, y_pred[:, 1])

