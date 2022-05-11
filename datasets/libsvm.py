from benchopt import BaseDataset

from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from libsvmdata import fetch_libsvm


class Dataset(BaseDataset):

    name = "libsvm"

    parameters = {
        'dataset': ["bodyfat", "leukemia", "rcv1.binary"],
    }

    install_cmd = 'conda'
    requirements = ['pip:libsvmdata']

    def __init__(self, dataset="bodyfat"):
        self.dataset = dataset

    def get_data(self):

        X, y = fetch_libsvm(self.dataset)

        data = dict(X=X, y=y)

        return data
