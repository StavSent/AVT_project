from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, precision_score
import pickle
import numpy as np
import pandas as pd

data_folder = './results'
model_folder = './models'
bonafide_train_results_fname = '/bonafide_train_scores.npy'
spoof_train_results_fname = '/spoofed_train_scores.npy'
labels_train_fname = '/labels_train.npy'

def load_results():
    # Load Bonafide train results
    with open(data_folder + bonafide_train_results_fname, 'rb') as tf:
        bonafide_train_results = np.load(tf)

    # Load Spoof train results
    with open(data_folder + spoof_train_results_fname, 'rb') as tf:
        spoofed_train_results = np.load(tf)

    # Load train labels for initial training purposes
    with open(data_folder + labels_train_fname, 'rb') as tf:
        labels = np.load(tf)

    # 2D arrays stacked
    train_X = np.stack((bonafide_train_results, spoofed_train_results), axis=-1)
    train_Y = labels

    return train_X, train_Y


if __name__ == '__main__':
    # Load results
    (train_X, train_Y) = load_results()

    # Train SVM
    model = svm.SVC()
    model.fit(train_X, train_Y)

    # Save fitted model
    with open(model_folder + '/SVM.pkl', "wb") as f:
        pickle.dump(model, f)


