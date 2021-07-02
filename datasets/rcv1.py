from benchopt import BaseDataset

from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from libsvmdata import fetch_libsvm


class Dataset(BaseDataset):

    name = "rcv1"

    install_cmd = 'pip'
    requirements = ['libsvmdata']

    def get_data(self):

        X, y = fetch_libsvm("rcv1.binary")

        data = dict(X=X, y=y)

        return X.shape[1], data
