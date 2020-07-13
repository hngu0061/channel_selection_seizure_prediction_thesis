import pdb
import numpy as np
import os
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
import random
import csv
import matplotlib as plt
import pickle
import json


settings = json.load(open('SETTINGS.json'))
pat = settings['pat']