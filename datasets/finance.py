from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    # Dependencies of download_libsvm are scikit-learn, download and tqdm
    from libsvmdata import fetch_libsvm


class Dataset(BaseDataset):
    """S. Kogan, D Levin, BR. Routledge, JS. Sagiand and NA. Smith,
    'Predicting risk from financial reports with regression'.
    In Proceedings of the North American Association for Computational
    Linguistics Human Language Technologies Conference (2009).
    """
    name = "finance"
    is_sparse = True

    install_cmd = 'conda'
    requirements = ['pip:libsvmdata']

    def __init__(self):
        self.X, self.y = None, None

    def get_data(self):

        if self.X is None:
            self.X, self.y = fetch_libsvm('finance')

        data = dict(X=self.X, y=self.y)

        return data
