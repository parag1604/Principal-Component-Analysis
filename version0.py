"""
version0.py: PCA from scratch with an eigen-decomposition of the covariance.

What this file teaches
- Centre the data (subtract the mean of every feature). PCA is about variance
  around the mean; without centering the first "component" mostly points at
  the mean itself.
- The principal directions are the eigenvectors of the covariance matrix
  S = A_c^T A_c / N, and the eigenvalues are the variances along them.
- Explained variance ratio: lambda_i / sum_j lambda_j tells how much of the
  spread each direction captures.
- Projecting onto the top-2 directions gives a 2-D picture of the data.

This is the first file.
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

from utils import get_data, sorted_eigh, add_data_args


def main():
    parser = argparse.ArgumentParser(description="PCA via the covariance eigen-decomposition")
    add_data_args(parser)
    args = parser.parse_args()

    A, y = get_data(args.data, args.path)
    N, D = A.shape
    print(f"data: {N} samples x {D} features")

    # 1. centre
    mean = A.mean(axis=0)
    A_c = A - mean

    # 2. covariance (D x D) and its eigen-decomposition
    S = A_c.T @ A_c / N
    eigenvalues, eigenvectors = sorted_eigh(S)

    # 3. explained variance
    ratio = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(ratio)
    for k in (1, 2, 5, 10, 20):
        if k <= D:
            print(f"top {k:>2} components explain {100 * cumulative[k - 1]:.1f}% of the variance")

    # 4. project onto the top-2 principal directions
    Z = A_c @ eigenvectors[:, :2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    m = min(D, 50)
    ax1.bar(range(1, m + 1), ratio[:m], label="per component")
    ax1.plot(range(1, m + 1), cumulative[:m], color="black", marker=".", label="cumulative")
    ax1.set_xlabel("principal component")
    ax1.set_ylabel("explained variance ratio")
    ax1.legend()
    sc = ax2.scatter(Z[:, 0], Z[:, 1], c=y, s=8, cmap="tab10")
    ax2.set_xlabel("PC 1")
    ax2.set_ylabel("PC 2")
    ax2.set_title("Data projected on the top-2 components")
    plt.colorbar(sc, ax=ax2, label="label")
    plt.tight_layout()
    plt.savefig(f"pca_{args.data}.png")
    print(f"Saved pca_{args.data}.png")


if __name__ == "__main__":
    main()
