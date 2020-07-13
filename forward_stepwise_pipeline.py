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



# load the feature files and store it inside a dictionary for evaluation and concatenation
def get_feature_files(patient:int, model:int, type_of_date: str):
    pass




def main():
    toggle = True
    while toggle:
        patient_num = int(input('Please enter the patient number (1,2,3) you want to investigate: '))
        if patient_num in [1,2,3]:
            toggle = False
        else:
            print('Wrong input, please choose again')
    
    toggle = True
    while toggle:
        print('Available models:\n1.{}\n2.{}\n3.{}\n4.{}'.format('Extra Tree', 'Logistic Regression', 'Random Forest', 'LDA'))
        type_of_model = int(input('Please enter 1,2,3,4 to choose the model to investigate: '))
        if type_of_model in [1,2,3,4]:
            toggle = False
        else:
            print('Wrong input, please choose again')
    
    
if __name__ == '__main__':
    main()