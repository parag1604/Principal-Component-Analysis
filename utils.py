import os

import numpy as np

# the colon cancer CSV ships with the repository, next to this file
COLON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "colon_cancer.csv")


def get_data(name: str = "digits", path: str = None, seed: int = 0):
    """Return a data matrix A (N samples x D features) and labels y (or None).

    - "digits": sklearn's bundled 8x8 handwritten digits (1797 x 64), no download.
    - "colon":  the Alon et al. (1999) colon cancer gene-expression data
                (62 tissues x 2000 genes), included as data/colon_cancer.csv (or
                read from `path`): a header row, then one row per tissue with an
                id in the first column, 2000 expression values, and the class
                label in the last column (1 = tumour, 0 = normal).
    - "wide":   a synthetic stand-in with the same shape as "colon" (62 x 2000):
                a rank-5 signal plus noise. Useful when the CSV is not available.
    """
    if name == "digits":
        from sklearn.datasets import load_digits
        digits = load_digits()
        return digits.data.astype(float), digits.target
    if name == "colon":
        path = path or COLON_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found; see the Data section of the README")
        table = np.loadtxt(path, delimiter=",", skiprows=1)
        return table[:, 1:-1], table[:, -1]
    if name == "wide":
        rng = np.random.default_rng(seed)
        y = rng.integers(2, size=62)
        signal = rng.normal(size=(62, 5)) @ rng.normal(size=(5, 2000)) + 2.0 * y[:, None] * rng.normal(size=2000)
        return signal + 0.5 * rng.normal(size=(62, 2000)), y
    raise ValueError(f"unknown dataset {name}")


def sorted_eigh(S: np.ndarray, k: int = None):
    """Eigen-decomposition of a symmetric matrix, eigenvalues in DESCENDING order.

    scipy's eigh returns them in ascending order, so we flip. With k, only the
    top-k pairs are computed (much cheaper when k << D).
    """
    import scipy.linalg as la
    D = S.shape[0]
    if k is None:
        eigenvalues, eigenvectors = la.eigh(S)
    else:
        eigenvalues, eigenvectors = la.eigh(S, subset_by_index=(D - k, D - 1))
    idxs = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idxs], eigenvectors[:, idxs]


def add_data_args(parser):
    parser.add_argument("--data", default="digits", choices=["digits", "colon", "wide"],
                        help="digits (bundled with sklearn), colon (included in data/colon_cancer.csv), "
                             "or wide (synthetic, colon-shaped)")
    parser.add_argument("--path", default=None,
                        help="CSV path for --data colon (default: the included data/colon_cancer.csv)")
