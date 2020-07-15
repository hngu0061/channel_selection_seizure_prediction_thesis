import pdb
import numpy as np
import os
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
import random
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import csv
import matplotlib as plt
import pickle
import json


# Extra Tree classifier
def extra_tree(patient_num):
    if patient_num == 1:
        return ExtraTreesClassifier(n_estimators=3000, random_state=0, max_depth=11, n_jobs=2)
    elif patient_num == 2:
        return ExtraTreesClassifier(n_estimators=5000, random_state=0, max_depth=15, n_jobs=2,criterion='entropy')
    elif patient_num == 3:
        return ExtraTreesClassifier(n_estimators=4500, random_state=0, max_depth=15,criterion='entropy', n_jobs=2)

# Random Forest
def random_forest(patient_num):
    if patient_num == 1:
        return RandomForestClassifier(n_estimators=5000, random_state=0, max_depth=15, n_jobs=2,criterion='gini', min_samples_split=7)
    elif patient_num == 2:
        return RandomForestClassifier(n_estimators=5000, random_state=0, max_depth=15, n_jobs=2,criterion='gini', min_samples_split=7)
    elif patient_num == 3:
        return RandomForestClassifier(n_estimators=4500, random_state=0, max_depth=15,criterion='gini', n_jobs=2,min_samples_split=7)

# function to train model and generate AUC:
def predict_seizure(model_type: int, patient_num: int, train_data, test_data, test_labels):

    data = train_data
    test = test_data

    # clean the training data by removing nans
    data.dropna(thresh=data.shape[1]-3, inplace=True)

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    test.replace([np.inf, -np.inf], np.nan, inplace=True)

    data.fillna(0, inplace=True)
    test.fillna(0, inplace=True)

    data_file = data.File.values
    test_file = test.File.values

    # get labels
    labela=[int(((str(os.path.basename(n)).split('_'))[2]).split('.')[0]) for n in data_file]
    labelt=[int(((str(os.path.basename(n)).split('_'))[2]).split('.')[0]) for n in test_file]

    data['L'] = labela
    test['L'] = labelt

    data.sort_values(['L'], inplace=True, ascending=False)
    test.sort_values(['L'], inplace=True, ascending=False)

    labela = data.L.values
    labelt = test.L.values

    data_feat = data.drop(['File', 'pat', 'L'], axis=1)
    test_feat = test.drop(['File', 'pat', 'L'], axis=1)
    data_feat = data_feat.values
    test_feat = test_feat.values

    # generate model for classification
    if model_type == 1:
        model = extra_tree(patient_num)
    elif model_type == 2:
        model = LogisticRegression(max_iter=2000)
    elif model_type == 3:
        model = random_forest(patient_num)
    elif model_type == 4:
        model = LinearDiscriminantAnalysis()

    model.fit(data_feat, labela)
    y_pred = model.predict_proba(test_feat)
    predict_values = y_pred[:,1]
    data_predict = zip(test_file, predict_values)
    column_names = ['file_path', 'probability']

    data_predict = pd.DataFrame(data_predict, columns=column_names)
    data_predict['image'] = data_predict.apply(lambda row: row.file_path[:-4], axis=1)

    result_data = pd.merge(test_labels, data_predict, on='image')
    new_result_df = result_data[~result_data['usage'].str.contains('Ignored', na=False)]

    # return AUC value
    return metrics.roc_auc_score(new_result_df['class'], new_result_df['probability'])


# function to train model and generate AUC:
def predict_seizure_no_label(model_type: int, patient_num: int, train_data, test_data):

    data = train_data
    test = test_data

    # clean the training data by removing nans
    data.dropna(thresh=data.shape[1]-3, inplace=True)

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    test.replace([np.inf, -np.inf], np.nan, inplace=True)

    data.fillna(0, inplace=True)
    test.fillna(0, inplace=True)

    data_file = data.File.values
    test_file = test.File.values

    # get labels
    labela=[int(((str(os.path.basename(n)).split('_'))[2]).split('.')[0]) for n in data_file]
    labelt=[int(((str(os.path.basename(n)).split('_'))[2]).split('.')[0]) for n in test_file]

    data['L'] = labela
    test['L'] = labelt

    data.sort_values(['L'], inplace=True, ascending=False)
    test.sort_values(['L'], inplace=True, ascending=False)

    labela = data.L.values
    labelt = test.L.values

    data_feat = data.drop(['File', 'pat', 'L'], axis=1)
    test_feat = test.drop(['File', 'pat', 'L'], axis=1)
    data_feat = data_feat.values
    test_feat = test_feat.values

    # generate model for classification
    if model_type == 1:
        model = extra_tree(patient_num)
    elif model_type == 2:
        model = LogisticRegression(max_iter=2000)
    elif model_type == 3:
        model = random_forest(patient_num)
    elif model_type == 4:
        model = LinearDiscriminantAnalysis()

    model.fit(data_feat, labela)
    y_pred = model.predict_proba(test_feat)

    # return AUC value
    return metrics.roc_auc_score(labelt, y_pred[:,1])
