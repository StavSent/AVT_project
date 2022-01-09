from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, precision_score
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_folder = './results'
model_folder = './models'
bonafide_dev_results_fname = '/bonafide_dev_scores.npy'
spoof_dev_results_fname = '/spoofed_dev_scores.npy'
bonafide_results_fname = '/bonafide_scores.npy'
spoof_results_fname = '/spoofed_scores.npy'
labeles_dev_fname = '/labels_dev.npy'

# Load Bonafide dev results
with open(data_folder + bonafide_dev_results_fname, 'rb') as tf:
    bonafide_dev_results = np.load(tf)

# Load Spoof dev results
with open(data_folder + spoof_dev_results_fname, 'rb') as tf:
    spoofed_dev_results = np.load(tf)

# Load dev labels for initial training purposes
with open(data_folder + labeles_dev_fname, 'rb') as tf:
    labels = np.load(tf)

# 2D arrays stacked
train_X = np.stack((bonafide_dev_results, spoofed_dev_results), axis=-1)
train_Y = labels

# SVM
model = svm.SVC()
model.fit(train_X, train_Y)

with open(model_folder + '/SVM.pkl', "wb") as f:
    pickle.dump(model, f)


