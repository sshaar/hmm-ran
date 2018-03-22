from hmmlearn import hmm
import numpy as np




data = np.load('../dataset/train.npz')

feat = data['feat']
num = feat.shape[0]
seq_len = feat.shape[1]

feat = np.concatenate(feat, axis=0)

print 'Loaded data'

GMMHMM = hmm.GMMHMM(n_components=5, n_mix=32, algorithm='viterbi',
                            covariance_type='diag', n_iter=10, tol=0.01,
                            verbose=True)

GMMHMM.fit(feat, [seq_len]*num)

print GMMHMM.predict_proba(feat[:10, :])


from hmm.continuous.GMHMM import GMHMM
