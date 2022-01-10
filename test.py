from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_curve, auc
import numpy as np
import pickle

model_folder = './models'
results_folder = './results'

def fetch_results(train=False):
    correct_drct = '_train' if (train==True) else '_eval'

    # Open files containing results and SVM model
    with open(model_folder + '/SVM.pkl', "rb") as f:
        model = pickle.load(f)

    with open(results_folder + '/bonafide' + correct_drct + '_scores.npy', "rb") as f:
        bonafide_results = np.load(f)

    with open(results_folder + '/spoofed' + correct_drct + '_scores.npy', "rb") as f:
        spoof_results = np.load(f)

    with open(results_folder + '/labels' + correct_drct + '.npy', "rb") as f:
        labels = np.load(f)


    train_X = np.stack((bonafide_results, spoof_results), axis=-1)
    train_Y = labels
    predictions = model.predict(train_X)
    
    return (labels, predictions)


if __name__ == '__main__':
    # If true calculates metrics for 'train' dataset, else for 'dev'
    (labels, predictions) = fetch_results(False)

    print('Accuracy: ', accuracy_score(labels, predictions))
    print('Recall: ', recall_score(labels, predictions))
    print('Precision: ', precision_score(labels, predictions))
