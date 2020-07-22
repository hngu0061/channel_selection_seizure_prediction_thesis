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
import pickle
import json
import predict_seizure as ps


def main():
    feat = json.load(open("SETTINGS.json"))

    # Get the ordered channel
    channel_set = feat.ordered_channel
    # Get all the file names
    filenames = []
    for i in range(16):
        filenames = filenames.append(i)
    best_auc_score = 0
    eliminated = 0

    print("Backward elimation:\n\tCurrent AUC score: {}".format(best_auc_score))
    for i in range(len(channel_set)):
        # Get current auc score
        current_channel_set = channel_set[i + 1 :]
        current_auc_score = 0

        # merge individual file into one file:

        print(
            "\tEliminating channel: {} \n\tCurrent AUC score: {}".format(
                channel_set[i], current_auc_score
            )
        )

        if current_auc_score >= best_auc_score:
            best_auc_score = current_auc_score
        else:
            print("\tStop eliminating channels")

            break

    print("Best channel set:")


if __name__ == "__main__":
    main()

