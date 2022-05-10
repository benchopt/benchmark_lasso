import numpy as np

from benchopt import BaseDataset


class Dataset(BaseDataset):

    name = "Pathological"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'n_samples': [
            11,
            4
        ],
    }

    def __init__(self, n_samples=11, random_state=27):
        # Store the parameters of the dataset
        self.n_samples = n_samples
        self.n_features = n_samples
        self.rng = np.random.RandomState(random_state)
        self.diago = self.rng.rand(n_samples)

    def get_data(self):
        y = np.ones(self.n_samples)
        tri = 2 * np.triu(np.ones([self.n_samples, self.n_samples]))
        print(self.diago.shape)
        X = tri * self.diago - np.diag(self.diago)
        data = dict(X=X, y=y)

        return self.n_features, data
