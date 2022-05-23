from benchopt import BaseDataset
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from sklearn.datasets import fetch_openml


class Dataset(BaseDataset):
    name = "MEG"
    install_cmd = "conda"
    requirements = ["scikit-learn"]

    @staticmethod
    def _load_meg_data(condition="Left Auditory"):
        dataset = fetch_openml(data_id=43884)
        X = dataset.data[:, :7498]

        if condition == "Left Auditory":
            idx = XXX
        else:
            idx = XXX
        y = dataset.data[:, idx]
        return X, y

    def get_data(self):
        try:
            X, y = self.X, self.y
        except AttributeError:
            X, y = self._load_meg_data()
            self.X, self.y = X, y
        return dict(X=X, y=y)
