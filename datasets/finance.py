from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    # Dependencies of download_libsvm are scikit-learn, download and tqdm
    from libsvmdata import fetch_libsvm


class Dataset(BaseDataset):
    name = "finance"
    is_sparse = True

    install_cmd = 'conda'
    requirements = ['pip:libsvmdata']

    def get_data(self):

        X, y = fetch_libsvm('finance')
        data = dict(X=X, y=y)

        return X.shape[1], data
