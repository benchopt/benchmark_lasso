from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import pooch  # noqa: F401
    import numpy as np
    from scipy.datasets import ascent
    from scipy.fftpack import dctn, idctn
    from scipy.sparse.linalg import LinearOperator


class Dataset(BaseDataset):
    name = "dct_inpainting"
    requirements = ["pooch"]

    def __init__(self, random_state=27):
        self.random_state = random_state
        self.X, self.y = None, None

    @staticmethod
    def _load_image(noise_level=0.01, random_state=27):
        rng = np.random.RandomState(random_state)

        y = ascent().astype(np.float64) / 255.0
        y += noise_level * rng.normal(0, 1.0, size=y.shape)

        return y

    def get_data(self):
        if self.X is None or self.y is None:
            y = self._load_image(noise_level=0.1,
                                 random_state=self.random_state)
            pw, ph = y.shape

            rng = np.random.RandomState(self.random_state)
            mask = rng.binomial(1, 0.9, pw * ph).astype(bool)

            y = y.flatten()
            y[mask] = 0.0

            def _fwd_operator(u):
                dct_coeffs = idctn(u.reshape((pw, ph)), norm="ortho").reshape(
                    (pw * ph, ))
                dct_coeffs[mask] = 0.0
                return dct_coeffs

            def _rfwd_operator(x):
                masked_coeffs = x.copy()
                masked_coeffs[mask] = 0.0
                idct_coeffs = dctn(masked_coeffs.reshape((pw, ph)),
                                   norm="ortho").reshape((pw * ph, ))
                return idct_coeffs

            _fwd_scipy_operator = LinearOperator(
                (pw * ph, pw * ph),
                matvec=_fwd_operator,
                rmatvec=_rfwd_operator,
            )

            self.X, self.y = _fwd_scipy_operator, y
        return dict(X=self.X, y=self.y)
