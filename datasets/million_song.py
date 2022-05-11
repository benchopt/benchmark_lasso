from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import os, tarfile, requests, h5py
    import numpy as np


class Dataset(BaseDataset):
    name = "million_song"

    references = [
        "Thierry Bertin-Mahieux, Daniel P.W. Ellis, Brian Whitman, and Paul Lamere,"
        "The Million Song Dataset. In Proceedings of the 12th International Society,"
        "for Music Information Retrieval Conference (ISMIR 2011), 2011."
    ]

    def get_data(self):
        X_path, y_path = "msd_X.npy", "msd_y.npy"
        if not (os.path.exists(X_path) and os.path.exists(y_path)):
            self._download_data()
            self._generate_dataset()
        return self._read_data(X_path, y_path)

    def _read_data(self, X_path, y_path):
        X = np.load(X_path)
        y = np.load(y_path)
        return dict(X=X, y=y)

    def _download_data(self):
        url = "http://labrosa.ee.columbia.edu/~dpwe/tmp/millionsongsubset.tar.gz"
        out_path = "millionsongsubset.tar.gz"

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(out_path, 'wb') as f:
                f.write(response.raw.read())
        else:
            raise Exception("An unexpected error occurred while downloading data. " +
                            "Status code: %s" % response.status_code)

        with tarfile.open(out_path) as tar_file:
            tar_file.extractall()

        # Clean
        os.remove(out_path)

    def _generate_dataset(self):
        pass


