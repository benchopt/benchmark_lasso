from benchopt import BaseDataset
from benchopt import safe_import_context


with safe_import_context() as ctx:
    import mne
    import numpy as np
    from mne.datasets import sample
    from mne.inverse_sparse.mxne_inverse import _prepare_gain
    from mne.utils import logger
    logger.setLevel(0)


class Dataset(BaseDataset):
    name = "MEG"
    install_cmd = "conda"
    requirements = ["mne"]

    @staticmethod
    def _load_meg_data(condition="Left Auditory"):
        data_path = sample.data_path() + '/MEG/sample'
        loose = 0
        depth = 0.8

        fwd_fname = data_path + '/sample_audvis-meg-eeg-oct-6-fwd.fif'
        ave_fname = data_path + '/sample_audvis-ave.fif'
        cov_fname = data_path + '/sample_audvis-shrunk-cov.fif'

        # Read noise covariance matrix
        noise_cov = mne.read_cov(cov_fname)
        # Handling average file
        evoked = mne.read_evokeds(
            ave_fname, condition=condition, baseline=(None, 0))
        evoked.crop(tmin=0.04, tmax=0.18)

        evoked = evoked.pick_types(eeg=False, meg=True)
        # Handling forward solution
        forward = mne.read_forward_solution(fwd_fname)

        # Handle depth weighting and whitening (here is no weights)
        forward, gain, gain_info, whitener, _, _ = _prepare_gain(
            forward, evoked.info, noise_cov, pca=False, depth=depth,
            loose=loose, weights=None, weights_min=None, rank=None)

        # Select channels of interest
        sel = [evoked.ch_names.index(name) for name in gain_info['ch_names']]
        M = evoked.data[sel]

        # Whiten data
        M = whitener @ M
        tmin, tmax = evoked.tmin, evoked.tmax
        tpeak = evoked.get_peak(ch_type="grad")[1]
        peak_idx = (tpeak - tmin) / (tmax - tmin) * (evoked.data.shape[1] - 1)
        peak_idx = int(np.round(peak_idx))

        return gain, M[:, peak_idx]

    def get_data(self):
        try:
            X, y = self.X, self.y
        except AttributeError:
            X, y = self._load_meg_data()
            self.X, self.y = X, y
        return dict(X=X, y=y)
