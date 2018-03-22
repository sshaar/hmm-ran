from hmmlearn.hmm import GaussianHMM
import numpy as np


GMMHMM = GaussianHMM(n_components=25, n_mix=32, algorithm='viterbi',
                            covariance_type='diag', n_iter=15, tol=0.01,
                            verbose=True)

data = np.load('../../dataset/train.npz')

feat = data['feat']

seq_len = feat.shape[1]

feat = np.concatenate(feat, axis=0)

GMMHMM.fit(feat, seq_len)