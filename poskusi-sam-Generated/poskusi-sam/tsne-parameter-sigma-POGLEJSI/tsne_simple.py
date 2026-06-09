"""Poenostavljena različica t-SNE (za raziskovanje, ne za teste)."""
import numpy as np


def _pairwise_sq(X):
    s = np.sum(X ** 2, axis=1)
    D = s[:, None] + s[None, :] - 2 * X @ X.T
    return np.maximum(D, 0.0)


def tsne(X, dim=2, sigma=10.0, lr=200.0, epochs=500, seed=0):
    X = np.asarray(X, dtype=float)
    n = len(X)
    D = _pairwise_sq(X)
    P = np.exp(-D / (2 * sigma ** 2))
    np.fill_diagonal(P, 0.0)
    P = P / P.sum(axis=1, keepdims=True)
    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12)
    rng = np.random.default_rng(seed)
    Y = rng.normal(0, 1e-2, (n, dim))
    for _ in range(epochs):
        Dy = _pairwise_sq(Y)
        num = 1.0 / (1.0 + Dy)
        np.fill_diagonal(num, 0.0)
        Q = np.maximum(num / num.sum(), 1e-12)
        PQ = (P - Q) * num
        grad = 4.0 * ((np.diag(PQ.sum(axis=1)) - PQ) @ Y)
        Y = Y - lr * grad
    return Y
