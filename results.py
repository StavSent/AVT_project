from sklearn.mixture import GaussianMixture as GMM
import pickle
import clean_feature_extraction as cfe
import numpy as np
import pandas as pd

data_folder = './data/LA'
model_folder = './models'
bonafide_fname = '/GMM_bonafide.pkl'
spoof_fname = '/GMM_spoof.pkl'

# Load Bonafide GMM model
bonafide_model = GMM(covariance_type='diag')
with open(model_folder + bonafide_fname, 'rb') as tf:
    bonafide_model._set_parameters(pickle.load(tf))

# Load Spoof GMM model
spoof_model = GMM(covariance_type='diag')
with open(model_folder + spoof_fname, 'rb') as tf:
    spoof_model._set_parameters(pickle.load(tf))

def calculate_and_save_results(train=False):
    # Change 'train' to 'dev' if the model is trained on 'dev' dataset
    correct_drct = 'train' if (train==True) else 'eval'

    # Calculate results for dev data
    eval_data_folder = data_folder + '/ASVspoof2019_LA_' + correct_drct + '/flac/'

    pd_files = pd.read_csv(data_folder + '/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.' + correct_drct + '.trl.txt', sep=' ', header=None)
    nrows = len(pd_files)

    predictions = np.zeros(nrows)
    labels = np.zeros(nrows)
    bonafide_scores = np.zeros(nrows)
    spoof_scores = np.zeros(nrows)
    index = 0

    for file in pd_files.values:
        Tx = cfe.extract_file_features(eval_data_folder + file[1] + '.flac')
        bonafide_score = bonafide_model.score(Tx.T)
        spoof_score = spoof_model.score(Tx.T)
        prediction = 1 if (bonafide_score - spoof_score > 0) else 0
        label = 1 if (file[4] == 'bonafide') else 0
        bonafide_scores[index] = bonafide_score
        spoof_scores[index] = spoof_score
        predictions[index] = prediction
        labels[index] = label
        index+=1

        # # Uncomment if you want to see progress
        # print(index / nrows * 100)

    # Save results
    correct_results_drct = '_train' if (train==True) else '_eval'
    with open('./results/bonafide' + correct_results_drct + '_scores.npy', 'wb') as f:
        np.save(f, bonafide_scores)

    with open('./results/spoofed' + correct_results_drct + '_scores.npy', 'wb') as f:
        np.save(f, spoof_scores)

    with open('./results/labels' + correct_drct + '.npy', 'wb') as f:
        np.save(f, labels)


if __name__ == '__main__':
    # If true calculates results for 'train' dataset, else for 'dev'
    calculate_and_save_results(True)

    


