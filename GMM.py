import pandas as pd
import os
from sklearn.mixture import GaussianMixture as GMM
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score
import clean_feature_extraction as cfe
import time
from sklearn.preprocessing import StandardScaler as scaler
from numpy import log, exp, infty, zeros_like, errstate, finfo, sqrt
from scipy.special import logsumexp
from os.path import exists
import logging
import pickle

data_file = './features/dev_features.pkl'
init_file = './features/init_dev_features.pkl'

train_folder = './data/LA/ASVspoof2019_LA_dev/flac/'
train_key = './data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'

model_folder = './models/'
bonafide_fname = 'GMM_bonafide.pkl'
spoof_fname = 'GMM_spoof.pkl'

feature_names = ['mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10',
                 'mfcc_11', 'mfcc_12', 'mfcc_13', 'delta_mfcc_1', 'delta_mfcc_2', 'delta_mfcc_3', 'delta_mfcc_4',
                 'delta_mfcc_5', 'delta_mfcc_6', 'delta_mfcc_7', 'delta_mfcc_8', 'delta_mfcc_9', 'delta_mfcc_10',
                 'delta_mfcc_11', 'delta_mfcc_12', 'delta_mfcc_13', 'delta_mfcc2_1', 'delta_mfcc2_2', 'delta_mfcc2_3',
                 'delta_mfcc2_4', 'delta_mfcc2_5', 'delta_mfcc2_6', 'delta_mfcc2_7', 'delta_mfcc2_8', 'delta_mfcc2_9',
                 'delta_mfcc2_10', 'delta_mfcc2_11', 'delta_mfcc2_12', 'delta_mfcc2_13', 'spectral_centroid',
                 'spectral_rolloff', 'spectral_flatness', 'zero_crossing_rate']


def main(do_scaling=False):
    start = time.time()

    if not features_exist(init=True):
        cfe.extract_init_features()
    init_data = pd.read_pickle(init_file)
    bonafide_X = init_data.loc[init_data['label'] == 1, feature_names]
    spoof_X = init_data.loc[init_data['label'] == 0, feature_names]
    del init_data
    if do_scaling:
        sc = scaler()
        bonafide_X = sc.fit(bonafide_X).transform(bonafide_X)
        spoof_X = sc.fit(spoof_X).transform(spoof_X)
    print("Init Data Loaded ", time_from(start))

    bonafide_gmm = train_gmm(X=bonafide_X, bonafide=True)
    spoof_gmm = train_gmm(X=spoof_X, bonafide=False)

    # if not features_exist():
    #     cfe.extract_features()
    # full_data = pd.read_pickle(data_file)
    # bonafide_X = full_data.loc[full_data['label'] == 1, feature_names]
    # spoof_X = full_data.loc[full_data['label'] == 0, feature_names]
    # if do_scaling:
    #     sc = scaler()
    #     bonafide_X = sc.fit(bonafide_X).transform(bonafide_X)
    #     spoof_X = sc.fit(spoof_X).transform(spoof_X)
    # print("Full Data Loaded ", time_from(start))

    print('Init Done')

    bonafide_gmm = train_gmm(X=bonafide_X, bonafide=True, gmm=bonafide_gmm)
    spoof_gmm = train_gmm(X=spoof_X, bonafide=False, gmm=spoof_gmm)

    # score(bonafide=bonafide_gmm, spoof=spoof_gmm, data=full_data)
    # print("Finished ", time_from(start))


def train_gmm(X, bonafide, gmm=None):
    start = time.time()
    init = gmm is None
    data_label = "bonafide" if bonafide else "spoof"
    if not model_is_trained(init):
        component_count = 512
        if init:
            model = GMM(n_components=component_count,
                        covariance_type='diag',
                        max_iter=10,
                        verbose=2,
                        verbose_interval=1
                        ).fit(X=X)
            if bonafide:
                with open(model_folder + "init_" + bonafide_fname, "wb") as f:
                    pickle.dump(model._get_parameters(), f)
                print("Bonafide Model initialized ", time_from(start))
            else:
                with open(model_folder + "init_" + spoof_fname, "wb") as f:
                    pickle.dump(model._get_parameters(), f)
                print("Bonafide Model initialized ", time_from(start))
        else:
            prev_lower_bound = -infty
            model_file = bonafide_fname if bonafide else spoof_fname
            full_gmm_file = model_folder + model_file

            model = GMM(covariance_type='diag')
            model._set_parameters(gmm._get_parameters())

            for i in range(1):
                if exists(full_gmm_file):
                    model = GMM(covariance_type='diag')
                    with open(full_gmm_file, "rb") as tf:
                        model._set_parameters(pickle.load(tf))

                nk_acc = zeros_like(model.weights_)
                mu_acc = zeros_like(model.means_)
                sigma_acc = zeros_like(model.covariances_)
                log_prob_norm_acc = 0
                n_samples = 0
                pd_files = pd.read_csv(train_key, sep=' ', header=None)
                files = pd_files[pd_files[4] == data_label][1]

                for file in files.values:
                    Tx = cfe.extract_file_features(train_folder + file + ".flac")
                    n_samples += Tx.shape[1]

                    # e step
                    weighted_log_prob = model._estimate_weighted_log_prob(Tx.T)
                    log_prob_norm = logsumexp(weighted_log_prob, axis=1)
                    with errstate(under='ignore'):
                        # ignore underflow
                        log_resp = weighted_log_prob - log_prob_norm[:, None]
                    log_prob_norm_acc += log_prob_norm.sum()

                    # m step preparation
                    resp = exp(log_resp)
                    nk_acc += resp.sum(axis=0) + 10 * finfo(log(1).dtype).eps
                    mu_acc += resp.T @ Tx.T
                    sigma_acc += resp.T @ (Tx.T ** 2)

                # m step
                model.means_ = mu_acc / nk_acc[:, None]
                model.covariances_ = sigma_acc / nk_acc[:, None] - model.means_ ** 2 + model.reg_covar
                model.weights_ = nk_acc / n_samples
                model.weights_ /= model.weights_.sum()
                if (model.covariances_ <= 0.0).any():
                    raise ValueError("ill-defined empirical covariance")
                model.precisions_cholesky_ = 1. / sqrt(model.covariances_)

                with open(full_gmm_file, "wb") as f:
                    pickle.dump(model._get_parameters(), f)

                # infos
                lower_bound = log_prob_norm_acc / n_samples
                change = lower_bound - prev_lower_bound
                logging.info("  Iteration %d\t llh %.5f\t ll change %.5f" % (i, lower_bound, change))
                prev_lower_bound = lower_bound

                if abs(change) < model.tol:
                    logging.info('  Coverged; too small change')
                    model.converged_ = True
                    break
    else:
        if init:
            if bonafide:
                model = GMM(covariance_type='diag')
                with open(model_folder + 'init_' + bonafide_fname, "rb") as tf:
                    model._set_parameters(pickle.load(tf))
            else:
                model = GMM(covariance_type='diag')
                with open(model_folder + 'init_' + spoof_fname, "rb") as tf:
                    model._set_parameters(pickle.load(tf))
        else:
            if bonafide:
                model = GMM(covariance_type='diag')
                with open(model_folder + bonafide_fname, "rb") as tf:
                    model._set_parameters(pickle.load(tf))
            else:
                model = GMM(covariance_type='diag')
                with open(model_folder + spoof_fname, "rb") as tf:
                    model._set_parameters(pickle.load(tf))

    return model


def score(bonafide, spoof, data):
    scores = bonafide.score(data.loc[:, feature_names]) - spoof.score(data.loc[:, feature_names])
    pred = scores > 0
    acc = accuracy_score(data.loc['label'], pred)
    pre = precision_score(data.loc['label'], pred)
    rec = recall_score(data.loc['label'], pred)
    print("Accuracy Score: ", acc)
    print("Precision Score: ", pre)
    print("Recall Score: ", rec)


def time_from(t):
    res = time.time() - t
    h = int(res / 3600)
    m = int((res % 3600) / 60)
    s = int(res % 60)
    tm = str(s) + "s"
    if m > 1:
        tm = str(m) + "m " + tm
    if h > 1:
        tm = str(h) + "h " + tm
    return tm


def features_exist(init=False):
    if init:
        return os.path.exists(init_file)
    return os.path.exists(data_file)


def model_is_trained(init=False):
    if init:
        return os.path.exists(model_folder + 'init_' + bonafide_fname) and os.path.exists(
            model_folder + 'init_' + spoof_fname)
    return os.path.exists(model_folder + bonafide_fname) and os.path.exists(model_folder + spoof_fname)


if __name__ == "__main__":
    main()
