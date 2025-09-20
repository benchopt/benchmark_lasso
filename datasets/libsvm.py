from benchopt import BaseDataset

from libsvmdata import fetch_libsvm
from sklearn.preprocessing import StandardScaler


class Dataset(BaseDataset):

    name = "libsvm"

    parameters = {
        'dataset': ["bodyfat", "leukemia", "rcv1.binary", "YearPredictionMSD"],
    }

    install_cmd = 'conda'
    requirements = ['pip::libsvmdata']
    references = [
        "C. Chang and CJ. Lin, "
        "'ACM transactions on intelligent systems and technology (TIST)', "
        "Acm New York, USA vol 2 (2011)."
    ]

    def get_data(self):

        X, y = fetch_libsvm(self.dataset)

        if self.dataset == "YearPredictionMSD":
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        return dict(X=X, y=y)
