from clean_feature_extraction import extract_file_features
from sklearn.metrics import accuracy_score, recall_score, precision_score
import numpy as np
from librosa.feature import spectral
import pickle

model_folder = './models'
results_folder = './results'

with open(model_folder + '/SVM.pkl', "rb") as f:
    model = pickle.load(f)

with open(results_folder + '/bonafide_scores.npy', "rb") as f:
    bonafide_results = np.load(f)

with open(results_folder + '/spoofed_scores.npy', "rb") as f:
    spoof_results = np.load(f)

with open(results_folder + '/labels.npy', "rb") as f:
    labels = np.load(f)


train_X = np.stack((bonafide_results, spoof_results), axis=-1)
train_Y = labels

predictions = model.predict(train_X)

print('Accuracy: ', accuracy_score(labels, predictions))
print('Recall: ', recall_score(labels, predictions))
print('Precision: ', precision_score(labels, predictions))

