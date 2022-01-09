import librosa
import numpy as np
from librosa.feature import spectral
import pandas as pd
from pathlib import Path
import os

def generate_labels(tf):
    label_dict = {}
    while True:
        line = tf.readline()
        if not line:
            break
        if 'e' == line[-2]:
            label_dict[line[8:20]] = 1
        else:
            label_dict[line[8:20]] = 0
    return label_dict
    

protocol_dir = './data/LA/ASVspoof2019_LA_cm_protocols/'
dataset_dir = './data/LA/ASVspoof2019_LA_'
file_dir = '/flac/'
label_file = open(protocol_dir + 'ASVspoof2019.LA.cm.dev.trl.txt')
data_files = [p for p in Path().glob(dataset_dir + 'dev' + file_dir + 'LA_D_*.flac')]

labels = generate_labels(label_file)
n_fft, hop = 512, 256


def extract_init_features():
    print("Initialization Features Extraction...")

    total_files = len(data_files)
    progress_step = int(total_files/200)
    progress = 0

    # n_fft, hop = 512, 256

    features = np.empty((40 * len(data_files), 44), float)
    index = 0

    # labels = generate_labels(label_file)

    for file in data_files:
        if progress % 10 == 0:
            timeseries, samplerate = librosa.load(file, sr=None)

            spectral_centroid = spectral.spectral_centroid(y=timeseries, sr=samplerate, n_fft=n_fft, hop_length=hop)
            spectral_rolloff = spectral.spectral_rolloff(y=timeseries, sr=samplerate, hop_length=hop)
            spectral_flatness = spectral.spectral_flatness(y=timeseries, hop_length=hop)
            zero_crossing_rate = spectral.zero_crossing_rate(y=timeseries, frame_length=n_fft, hop_length=hop)

            mfcc = librosa.feature.mfcc(y=timeseries, sr=samplerate, n_mfcc=13, hop_length=hop)
            mfcc_delta = librosa.feature.delta(mfcc, order=1, mode='nearest')
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2, mode='nearest')
            mfccs = np.concatenate((mfcc.T, mfcc_delta.T, mfcc_delta2.T), axis=1)

            label = np.ones(zero_crossing_rate.shape)*labels[os.path.basename(file)[:-5]]
            feature_vector = np.concatenate((mfccs, spectral_centroid.T, spectral_flatness.T, spectral_rolloff.T,
                                             zero_crossing_rate.T, label.T), axis=1)

            new_index = index + feature_vector.shape[0]
            features[index:new_index] = feature_vector
            index = new_index

        progress = progress+1
        if progress%progress_step == 0:
            print("Progress: ", str(100 * progress / total_files)[:6])

    df = pd.DataFrame(features, columns=[
        'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7',
        'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13',
        'delta_mfcc_1', 'delta_mfcc_2', 'delta_mfcc_3', 'delta_mfcc_4', 'delta_mfcc_5',
        'delta_mfcc_6', 'delta_mfcc_7', 'delta_mfcc_8', 'delta_mfcc_9', 'delta_mfcc_10',
        'delta_mfcc_11', 'delta_mfcc_12', 'delta_mfcc_13',
        'delta_mfcc2_1', 'delta_mfcc2_2', 'delta_mfcc2_3', 'delta_mfcc2_4', 'delta_mfcc2_5',
        'delta_mfcc2_6', 'delta_mfcc2_7', 'delta_mfcc2_8', 'delta_mfcc2_9', 'delta_mfcc2_10',
        'delta_mfcc2_11', 'delta_mfcc2_12', 'delta_mfcc2_13',
        'spectral_centroid', 'spectral_rolloff', 'spectral_flatness', 'zero_crossing_rate', 'label'
    ])
    df = df.loc[~(df == 0).all(axis=1)]
    df.to_pickle("./features/init_dev_features.pkl")


def extract_features():
    print("Feature Extraction...")

    total_files = len(data_files)
    progress_step = int(total_files/200)
    progress = 0

    # n_fft, hop = 512, 256

    features = np.empty((400 * len(data_files), 44), float)
    index = 0

    # labels = generate_labels(label_file)

    for file in data_files:
        timeseries, samplerate = librosa.load(file, sr=None)

        spectral_centroid = spectral.spectral_centroid(y=timeseries, sr=samplerate, n_fft=n_fft, hop_length=hop)
        spectral_rolloff = spectral.spectral_rolloff(y=timeseries, sr=samplerate, hop_length=hop)
        spectral_flatness = spectral.spectral_flatness(y=timeseries, hop_length=hop)
        zero_crossing_rate = spectral.zero_crossing_rate(y=timeseries, frame_length=n_fft, hop_length=hop)

        mfcc = librosa.feature.mfcc(y=timeseries, sr=samplerate, n_mfcc=13, hop_length=hop)
        mfcc_delta = librosa.feature.delta(mfcc, order=1, mode='nearest')
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2, mode='nearest')
        mfccs = np.concatenate((mfcc.T, mfcc_delta.T, mfcc_delta2.T), axis=1)

        label = np.ones(zero_crossing_rate.shape)*labels[os.path.basename(file)[:-5]]
        feature_vector = np.concatenate((mfccs, spectral_centroid.T, spectral_flatness.T, spectral_rolloff.T,
                                         zero_crossing_rate.T, label.T), axis=1)

        new_index = index + feature_vector.shape[0]
        features[index:new_index] = feature_vector
        index = new_index

        progress = progress+1
        if progress%progress_step == 0:
            print("Progress: ", str(100 * progress / total_files)[:6])

    df = pd.DataFrame(features, columns=[
        'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7',
        'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13',
        'delta_mfcc_1', 'delta_mfcc_2', 'delta_mfcc_3', 'delta_mfcc_4', 'delta_mfcc_5',
        'delta_mfcc_6', 'delta_mfcc_7', 'delta_mfcc_8', 'delta_mfcc_9', 'delta_mfcc_10',
        'delta_mfcc_11', 'delta_mfcc_12', 'delta_mfcc_13',
        'delta_mfcc2_1', 'delta_mfcc2_2', 'delta_mfcc2_3', 'delta_mfcc2_4', 'delta_mfcc2_5',
        'delta_mfcc2_6', 'delta_mfcc2_7', 'delta_mfcc2_8', 'delta_mfcc2_9', 'delta_mfcc2_10',
        'delta_mfcc2_11', 'delta_mfcc2_12', 'delta_mfcc2_13',
        'spectral_centroid', 'spectral_rolloff', 'spectral_flatness', 'zero_crossing_rate', 'label'
    ])
    df = df.loc[~(df == 0).all(axis=1)]
    df.to_pickle("./features/dev_features.pkl")


def extract_file_features(file, labeled=False):
    timeseries, samplerate = librosa.load(file, sr=None)

    spectral_centroid = spectral.spectral_centroid(y=timeseries, sr=samplerate, n_fft=n_fft, hop_length=hop)
    spectral_rolloff = spectral.spectral_rolloff(y=timeseries, sr=samplerate, hop_length=hop)
    spectral_flatness = spectral.spectral_flatness(y=timeseries, hop_length=hop)
    zero_crossing_rate = spectral.zero_crossing_rate(y=timeseries, frame_length=n_fft, hop_length=hop)

    mfcc = librosa.feature.mfcc(y=timeseries, sr=samplerate, n_mfcc=13, hop_length=hop)
    mfcc_delta = librosa.feature.delta(mfcc, order=1, mode='nearest')
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2, mode='nearest')
    mfccs = np.concatenate((mfcc.T, mfcc_delta.T, mfcc_delta2.T), axis=1)

    if labeled: 
        label = np.ones(zero_crossing_rate.shape)*labels[os.path.basename(file)[:-5]]
        feature_vector = np.concatenate((mfccs, spectral_centroid.T, spectral_flatness.T, spectral_rolloff.T,
                                         zero_crossing_rate.T, label.T), axis=1)
        return feature_vector

    feature_vector = np.concatenate((mfccs, spectral_centroid.T, spectral_flatness.T, spectral_rolloff.T,
                                    zero_crossing_rate.T), axis=1)
    return feature_vector.T


if __name__ == "__main__":
    extract_init_features()
