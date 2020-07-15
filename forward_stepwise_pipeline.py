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
import predict_seizure as ps

# load the feature files and store it inside a dictionary for evaluation and concatenation
def get_feature_files(patient_num: int):
    settings = json.load(open("SETTINGS.json"))
    feature_training_dict = {}
    feature_testing_dict = {}

    for i in range(16):
        # creating dictionary for feature sets of training data
        # train = pd.read_csv(settings['feat']+'/Train/pat_'+str(patient_num)+'_long_newtrain_sub_' + str(i) + '.csv')
        train = pd.read_csv(
            settings["feat"] + "/pat_1_train_data_channel_" + str(i) + "_.csv",
            index_col=0,
        )
        feature_training_dict[i] = train

        # creating dictionary for feature sets of testing data
        # test = pd.read_csv(settings['feat']+'/Test/pat_'+str(patient_num)+'_long_newtest_sub_' + str(i) + '.csv')
        test = pd.read_csv(
            settings["feat"] + "/pat_1_test_data_channel_" + str(i) + "_.csv",
            index_col=0,
        )
        feature_testing_dict[i] = test

    return feature_training_dict, feature_testing_dict


i = 1
# function to merge two feature sets to one
def merge_features(channel_x, channel_y):
    # df = pd.merge(channel_x, channel_y, on=[ 'File', 'pat'])
    # global i
    # path = settings['feat'] + '/pat_1_merge_time+'
    data_x = channel_x.drop(["L"], axis=1)
    data_y = channel_y.drop(["L"], axis=1)
    return pd.merge(data_x, data_y, on=["File", "pat"])


def main():
    toggle = True
    while toggle:
        patient_num = int(
            input("Please enter the patient number (1,2,3) you want to investigate: ")
        )
        if patient_num in [1, 2, 3]:
            toggle = False
        else:
            print("Wrong input, please choose again")

    toggle = True
    while toggle:
        print(
            "Available models:\n1.{}\n2.{}\n3.{}\n4.{}".format(
                "Extra Tree", "Logistic Regression", "Random Forest", "LDA"
            )
        )
        type_of_model = int(
            input("Please enter 1,2,3,4 to choose the model to investigate: ")
        )
        if type_of_model in [1, 2, 3, 4]:
            toggle = False
        else:
            print("Wrong input, please choose again")

    # get labels for testing data
    # settings = json.load(open('SETTINGS.json'))
    # labelled_data = pd.read_csv(settings['feat'] + '/pat_' + str(patient_num) + '_test_data_labels.csv')

    # get all the features set
    train_feature_channel_dict, test_feature_channel_dict = get_feature_files(
        patient_num
    )

    # generate auc for each channel
    channel_auc_dict = {}
    for i in range(16):
        print("Evaluating channel {}...".format(i))
        # channel_auc_dict[i] = ps.predict_seizure(type_of_model, patient_num, train_feature_channel_dict[i], test_feature_channel_dict[i], labelled_data)
        channel_auc_dict[i] = ps.predict_seizure_no_label(
            type_of_model,
            patient_num,
            train_feature_channel_dict[i],
            test_feature_channel_dict[i],
        )

    # order auc_dict by descending order and convert to list
    ordered_channel_auc = [
        (k, v)
        for k, v in sorted(channel_auc_dict.items(), key=lambda x: x[1], reverse=True)
    ]

    print(ordered_channel_auc)

    print(
        "Best performance channel: {}\n\tAUC Score: {}".format(
            ordered_channel_auc[0][0], ordered_channel_auc[0][1]
        )
    )

    best_channels_so_far = [ordered_channel_auc[0][0]]
    best_auc_so_far = ordered_channel_auc[0][1]
    training_data_so_far = train_feature_channel_dict[ordered_channel_auc[0][0]]
    testing_data_so_far = test_feature_channel_dict[ordered_channel_auc[0][0]]

    # perform forward stepwise selection
    for i in range(1, 16):
        print(
            "Add channel {} for evaluating. Individual AUC score {}".format(
                ordered_channel_auc[i][0], ordered_channel_auc[i][1]
            )
        )
        temp_training_data_so_far = merge_features(
            training_data_so_far, train_feature_channel_dict[ordered_channel_auc[i][0]]
        )
        print(temp_training_data_so_far.columns)
        print(len(temp_training_data_so_far.columns))
        temp_testing_data_so_far = merge_features(
            testing_data_so_far, test_feature_channel_dict[ordered_channel_auc[i][0]]
        )
        print(temp_testing_data_so_far.columns)
        print(len(temp_testing_data_so_far.columns))
        # temp_auc = ps.predict_seizure(type_of_model, patient_num, temp_training_data_so_far, temp_testing_data_so_far, labelled_data)
        temp_auc = ps.predict_seizure_no_label(
            type_of_model,
            patient_num,
            temp_training_data_so_far,
            temp_testing_data_so_far,
        )
        if temp_auc >= best_auc_so_far:
            best_auc_so_far = temp_auc
            best_channels_so_far.append(ordered_channel_auc[i][0])
            training_data_so_far = temp_training_data_so_far
            testing_data_so_far = temp_testing_data_so_far
            print("Current AUC score: {}".format(best_auc_so_far))
        else:
            print("Temporary AUC score: {}".format(temp_auc))
            print(
                "AUC could not be improved when adding channel {}.".format(
                    ordered_channel_auc[i][0]
                )
            )
            break

    print(
        "Best channel sets: {}\n\tBest AUC Score:{}".format(
            best_channels_so_far, best_auc_so_far
        )
    )


if __name__ == "__main__":
    main()
