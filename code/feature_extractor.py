################################################################################
################################################################################
# This file is used to generate features from an audio file. It will collect all
#  training, testing and dev audios from all the different genres in the main
#  directory path.
################################################################################
################################################################################

import numpy as np
import librosa
import math
import re
import os

from torch.utils.data.dataset import Dataset

# Number of training instances per category.
TRAIN_NUM = 750
# Number of dev instances per category.
DEV_NUM = 100
# Number of test instances per category.
TEST_NUM = 150

# The main directory path that contains all the genre folders.
DIRECTORY_PATH = "/home/sabith/hmm-rnn/data/genres/"
# The name of genres in the DIRECTORY_PATH.
GENRE_LIST = ['blues/', 'classical/', 'country/', 'disco/', 'hiphop/', 'jazz/', 'metal/', 'pop/', 'reggae/']
# GENRE_LIST = ['classical/']


# This function takes in a directory path and return all the audio files of
#  .au format in a list.
def path_to_audiofiles(dir_folder):

    # This will contain all the audio files of .au format in dir_folder.
    list_of_audio = []

    # Iterate over all the files in the directory dir_folder.
    for file in os.listdir(dir_folder):

        # If the file is of the .au format append it to the audio list list_of_audio.
        if file.endswith(".au"):
            directory = "%s/%s" % (dir_folder, file)
            list_of_audio.append(directory)

    # return the list of all audio of format .au in dir_folder
    return list_of_audio

# This function extracts the features from a list of audio files. It will take
#  in the names of the audio files and returns a tuple (features, labels).
def extract_audio_features_list(list_of_audiofiles):

    # This is a magic number. The name came from number of samples taken and the
    #  hope length when features, e.g. MFCC, spectral_centroid, etc., are being
    #  extracted.
    timeseries_length = 130

    # These are constants for audio feature extractions. e.g. MFCC, etc.
    hop_length = 512

    # This is a holder to the data that will be extracted from the audio files.
    #  The shape is (num_audio_files x timeseries_length x num_feats) = (1000 x 1293 x 40)
    # This will be features that I will return later.
    data = np.zeros((len(list_of_audiofiles), timeseries_length, 40), dtype=np.float64)

    # True labels of the audio files
    target = []

    # Iterate over all audio file names.
    for i, file in enumerate(list_of_audiofiles):
        # Load the audio file using librosa, to get the frames.
        y, sr = librosa.load(file)

        # Extract MFCC features. the output will be a (20, timeseries_length).
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=20)

        # Extract spectral_centroid features. the output will be a (20, timeseries_length).
        spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)

        # Extract chroma_stft features. the output will be a (20, timeseries_length).
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)

        # Extract spectral_contrast features. the output will be a (20, timeseries_length).
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)

        # Get the genre from the audio file name.
        splits = re.split('[ .]', file)
        genre = re.split('[ /]', splits[0])[-1]
        target.append(genre)

        mfcc = mfcc.T
        spectral_center = spectral_center.T
        chroma = chroma.T
        spectral_contrast = spectral_contrast.T

        mfcc = np.pad(mfcc, ((0, timeseries_length-mfcc.shape[0]), (0, 0)), 'edge')
        spectral_center = np.pad(spectral_center, ((0, timeseries_length-spectral_center.shape[0]), (0, 0)), 'edge')
        chroma = np.pad(chroma, ((0, timeseries_length-chroma.shape[0]), (0, 0)), 'edge')
        spectral_contrast = np.pad(spectral_contrast, ((0, timeseries_length-spectral_contrast.shape[0]), (0, 0)), 'edge')

        # Place the received data in the data holder.
        data[i, :, 0:20] = mfcc[0:timeseries_length, :]
        data[i, :, 20:21] = spectral_center[0:timeseries_length, :]
        data[i, :, 21:33] = chroma[0:timeseries_length, :]
        data[i, :, 33:40] = spectral_contrast[0:timeseries_length, :]

        print("Extracted features audio track %i of %i." % (i + 1, len(list_of_audiofiles)))

    # return the tuple containing (features, labels) for all the audio files in
    #  list_of_audiofiles.
    return data, np.expand_dims(np.asarray(target), axis=1)


# Main function that will extract all audio features in a root directry directory_name.
# This function will return a dictionary that has dev, train and test entries.
#  Each value is a tuple ((num_audio_files x timeseries_length x 40), (num_audio_files, )).
def extract_audio_features(directory_name):

    train_feat = []
    train_label = []
    dev_feat = []
    dev_label = []
    test_feat = []
    test_label = []

    # Iterate over all the genres in the global variable GENRE_LIST.
    for genre in GENRE_LIST:

        genre_directory = DIRECTORY_PATH + genre + genre[:-1] + "_3"
        print ("Getting audio files from %s" %genre_directory)

        # Extract all the audio files in the directory genre_directory
        all_audiofile_genre = path_to_audiofiles(genre_directory)

        # Cut the all_audiofile_genre into 3 sets, one for train, dev and test.
        train_audiofile_genre = all_audiofile_genre[:TRAIN_NUM]
        dev_audiofile_genre = all_audiofile_genre[TRAIN_NUM:TRAIN_NUM+DEV_NUM]
        test_audiofile_genre = all_audiofile_genre[TRAIN_NUM+DEV_NUM:]

        # Extract the features from training set.
        train = extract_audio_features_list(train_audiofile_genre)
        # Extract the features from dev set.
        dev = extract_audio_features_list(dev_audiofile_genre)
        # Extract the features from testing set.
        test = extract_audio_features_list(test_audiofile_genre)

        # Append the extracted features of this genre to the list of all files.
        train_feat.extend(train[0])
        train_label.extend(train[1])

        # Append the extracted features of this genre to the list of all files.
        dev_feat.extend(dev[0])
        dev_label.extend(dev[1])

        # Append the extracted features of this genre to the list of all files.
        test_feat.extend(test[0])
        test_label.extend(test[1])

    # Convert to numpy array.
    train_feat = np.array(train_feat)
    train_label = np.array(train_label)

    # Convert to numpy array.
    dev_feat = np.array(dev_feat)
    dev_label = np.array(dev_label)

    # Convert to numpy array.
    test_feat = np.array(test_feat)
    test_label = np.array(test_label)

    print ("train_feat", train_feat.shape)
    print ("dev_feat", dev_feat.shape)
    print ("test_feat", test_feat.shape)

    # Return the dictionary. The values are tuples of each dataset. The first
    #  element is a 3D matrix and the second is 1D.
    # 3D matrix is the features for all audio files in subdirectories of directory_name.
    # 1D are the labels. It is an array of strings.
    return {"train": (train_feat, train_label), "dev": (dev_feat, dev_label), "test": (test_feat, test_label)}

data = extract_audio_features(DIRECTORY_PATH)

tri = data["train"]
np.savez("train.npz", feat=tri[0], target=tri[1])

tri = data["dev"]
np.savez("dev.npz", feat=tri[0], target=tri[1])

tri = data["test"]
np.savez("test.npz", feat=tri[0], target=tri[1])
