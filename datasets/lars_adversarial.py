import numpy as np

from benchopt import BaseDataset


class Dataset(BaseDataset):
    """Adversarial example of Mairal and Yu, 'Complexity Analysis
     of the Lasso Regularization Path', 2012."""
    name = "lars_adversarial"

    parameters = {
        'n_samples': [100],
    }

    def __init__(self, n_samples=10):
        self.n_samples = n_samples
        self.diago = np.logspace(-3, -8, n_samples) / n_samples

    def get_data(self):
        y = np.ones(self.n_samples)
        tri = 2 * np.triu(np.ones([self.n_samples, self.n_samples]))
        X = tri * self.diago - np.diag(self.diago)

        return dict(X=X, y=y)
