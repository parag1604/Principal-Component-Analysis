"""
version2.py: PCA through the SVD, and the N << D trick.

What is NEW compared to version1.py
- We never need the D x D covariance. With the thin SVD of the centred data,
  A_c = U Sigma V^T, the columns of V are the principal directions and the
  eigenvalues are lambda_i = sigma_i^2 / N. The SVD is also numerically safer:
  forming A_c^T A_c squares the condition number.
- When there are far fewer samples than features (gene expression: N = 62,
  D = 2000) the N x N Gram matrix A_c A_c^T has the same non-zero eigenvalues.
  Its eigenvectors u_i give the principal directions as v_i = A_c^T u_i / sigma_i.
- At most N - 1 eigenvalues are non-zero after centering, so for wide data PCA
  can keep at most N - 1 components.

We compute the top components all three ways, check they agree, and time them.
"""
import argparse
import time

import numpy as np

from utils import get_data, sorted_eigh, add_data_args


def pca_covariance(A_c, k):
    N = A_c.shape[0]
    return sorted_eigh(A_c.T @ A_c / N, k)        # D x D problem


def pca_svd(A_c, k):
    N = A_c.shape[0]
    U, s, Vt = np.linalg.svd(A_c, full_matrices=False)
    return s[:k] ** 2 / N, Vt[:k].T


def pca_gram(A_c, k):
    N = A_c.shape[0]
    eigenvalues, U = sorted_eigh(A_c @ A_c.T, k)  # N x N problem
    sigma = np.sqrt(np.maximum(eigenvalues, 1e-12))
    V = A_c.T @ U / sigma                          # map back to feature space
    return eigenvalues / N, V


def main():
    parser = argparse.ArgumentParser(description="PCA via covariance, SVD and Gram matrix")
    add_data_args(parser)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    A, _ = get_data(args.data, args.path)
    N, D = A.shape
    A_c = A - A.mean(axis=0)
    k = min(args.k, N - 1, D)
    print(f"data: {N} samples x {D} features, computing the top {k} components")

    results = {}
    for name, fn in [("covariance (D x D)", pca_covariance), ("SVD of the data", pca_svd),
                     ("Gram matrix (N x N)", pca_gram)]:
        start = time.perf_counter()
        eigenvalues, V = fn(A_c, k)
        elapsed = time.perf_counter() - start
        results[name] = (eigenvalues, V)
        print(f"{name:>20}: {1000 * elapsed:8.2f} ms, top eigenvalues {np.round(eigenvalues[:3], 3)}")

    ref_vals, ref_V = results["covariance (D x D)"]
    for name, (vals, V) in results.items():
        # eigenvectors are defined up to sign, so compare |v_ref . v|
        alignment = np.abs(np.sum(ref_V * V, axis=0)).min()
        print(f"{name:>20}: max eigenvalue difference {np.abs(vals - ref_vals).max():.2e}, "
              f"min |cosine| with covariance PCs {alignment:.6f}")

    rank = np.linalg.matrix_rank(A_c)
    print(f"rank of the centred data: {rank} (at most min(N - 1, D) = {min(N - 1, D)})")


if __name__ == "__main__":
    main()
