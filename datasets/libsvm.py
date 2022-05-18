from benchopt import BaseDataset

from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from libsvmdata import fetch_libsvm
    from sklearn.preprocessing import StandardScaler


class Dataset(BaseDataset):

    name = "libsvm"

    parameters = {
        'dataset': [
            "bodyfat", "leukemia", "news20.binary", "rcv1.binary",
            "YearPredictionMSD"],
    }

    install_cmd = 'conda'
    requirements = ['pip:git+https://github.com/mathurinm/libsvmdata@main']

    def __init__(self, dataset="bodyfat"):
        self.dataset = dataset

    def get_data(self):

        X, y = fetch_libsvm(self.dataset)

        if self.dataset == "YearPredictionMSD":
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            y -= y.mean()

        data = dict(X=X, y=y)

        return data
