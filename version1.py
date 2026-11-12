"""
version1.py: reconstruction error as a function of the number of components.

What is NEW compared to version0.py
- PCA as compression: keep the coordinates on the top-k directions,
  A_hat = A_c V_k, and map back, A_rec = A_hat V_k^T + mean.
- The reconstruction error falls as k grows, and it is exactly predicted by the
  eigenvalues we threw away (Eckart-Young):
      mean squared error = (1/D) * sum_{i>k} lambda_i.
  We print both numbers side by side to check this.
- Only the top-k eigenpairs are needed, so we ask eigh for just those.
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

from utils import get_data, sorted_eigh, add_data_args


def PCA(A, k, eigenvectors):
    # Get the top k eigenvectors
    top_k_eigenvectors = eigenvectors[:, :k]

    # Project the data
    return A @ top_k_eigenvectors, top_k_eigenvectors


def main():
    parser = argparse.ArgumentParser(description="PCA reconstruction error vs k")
    add_data_args(parser)
    parser.add_argument("--k", type=int, default=None, help="largest k to try (default: min(N, D))")
    args = parser.parse_args()

    A, _ = get_data(args.data, args.path)
    N, D = A.shape
    k = args.k or min(N, D)

    mean = A.mean(axis=0)
    A_c = A - mean
    S = A_c.T @ A_c / N
    total_variance = np.trace(S)

    # Get the top k eigenvalues and eigenvectors (descending)
    eigenvalues, eigenvectors = sorted_eigh(S, k)

    errors = []
    for i in range(1, k + 1):
        A_hat, eigen_basis = PCA(A_c, i, eigenvectors)
        A_reconstructed = A_hat @ eigen_basis.T + mean
        error = np.sqrt(np.mean((A - A_reconstructed) ** 2))
        predicted = np.sqrt(max(total_variance - eigenvalues[:i].sum(), 0) / D)
        errors.append(error)
        print("k = {}, RMSE = {}, predicted from eigenvalues = {}".format(
            i, np.round(error, 4), np.round(predicted, 4)))

    plt.plot(range(1, k + 1), errors)
    plt.xlabel("number of components k")
    plt.ylabel("reconstruction RMSE")
    plt.title(f"PCA reconstruction error ({args.data})")
    plt.tight_layout()
    plt.savefig(f"reconstruction_{args.data}.png")
    print(f"Saved reconstruction_{args.data}.png")

    if args.data == "digits":
        # show one digit reconstructed with a growing number of components
        ks = [1, 2, 5, 10, 20, k]
        fig, axes = plt.subplots(1, len(ks) + 1, figsize=(2 * (len(ks) + 1), 2.3))
        axes[0].imshow(A[0].reshape(8, 8), cmap="gray_r")
        axes[0].set_title("original")
        for ax, kk in zip(axes[1:], ks):
            V = eigenvectors[:, :kk]
            ax.imshow(((A_c[0] @ V) @ V.T + mean).reshape(8, 8), cmap="gray_r")
            ax.set_title(f"k = {kk}")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig("reconstructed_digit.png")
        print("Saved reconstructed_digit.png")


if __name__ == "__main__":
    np.random.seed(2023)
    main()
