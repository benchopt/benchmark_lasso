

import mne
import numpy as np
from mne.datasets import sample
from mne.inverse_sparse.mxne_inverse import _prepare_gain
# this script used MNE 1.0.2

condition = "Right Auditory"
data_path = sample.data_path() + '/MEG/sample'
loose = 0
depth = 0.8

fwd_fname = data_path + '/sample_audvis-meg-eeg-oct-6-fwd.fif'
ave_fname = data_path + '/sample_audvis-ave.fif'
cov_fname = data_path + '/sample_audvis-shrunk-cov.fif'

# Read noise covariance matrix
noise_cov = mne.read_cov(cov_fname)
# Handling forward solution
forward = mne.read_forward_solution(fwd_fname)
targets = {}

for condition in ["Left Auditory", "Right Auditory"]:
    evoked = mne.read_evokeds(
        ave_fname, condition=condition, baseline=(None, 0))
    evoked.crop(tmin=0.04, tmax=0.18)
    evoked = evoked.pick_types(eeg=False, meg=True)

    # Handle depth weighting and whitening (here is no weights)
    forward, gain, gain_info, whitener, _, _ = _prepare_gain(
        forward, evoked.info, noise_cov, pca=False, depth=depth,
        loose=loose, weights=None, weights_min=None, rank=None)

    # Select channels of interest
    sel = [evoked.ch_names.index(name) for name in gain_info['ch_names']]
    M = evoked.data[sel]

    # Whiten data
    M = whitener @ M
    targets[condition] = M

# gain is independent of condition:
data = np.hstack([gain, targets["Left Auditory"], targets["Right Auditory"]])
np.savetxt("data.csv", data, delimiter=',')
