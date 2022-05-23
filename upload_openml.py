import numpy as np

from openml.datasets.functions import create_dataset


name = "mne-sample-meg-auditory"
data = np.loadtxt("data.csv", delimiter=',')


descr = """
MEG data from auditory stimulation experiment using 305 sensors.
The design matrix/forward operator is `data[:, :7498]`.
The measurements for left stimulation are `data[:, 7498:7583]`.
The measurements for right stimulation are `data[:, 7583:]`.

The data was generated with the following script:
```
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
```
"""

citation = "Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier, D., Brodbeck, C., ... & Hamalainen, M. (2013). MEG and EEG data analysis with MNE-Python. Frontiers in neuroscience, 267."


attributes = (["X" + str(x).zfill(4) for x in range(7498)] +
              ["L" + str(x).zfill(2) for x in range(85)] +
              ["R" + str(x).zfill(2) for x in range(85)])

attributes = list(zip(attributes, ["REAL"] * data.shape[1]))


dataset = create_dataset(
    name=name,
    data=data,
    description=descr,
    contributor='Mathurin Massias',
    licence='BSD (from MNE)',
    citation=citation,
    creator="MNE contributors",
    collection_date="2022/05/23",
    language="English",
    attributes=attributes,
    default_target_attribute=None,
    ignore_attribute=None,
)

dataset.publish()
