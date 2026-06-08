import csv
import os

import numpy as np


def kmeans(X, k, max_iters=100, seed=0):
    """Metoda voditeljev (k-means). Vrne (centroids, labels)."""
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(max_iters):
        d = np.sqrt(((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2))
        labels = d.argmin(axis=1)
        newC = np.array([X[labels == j].mean(axis=0) if np.any(labels == j)
                         else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return C, labels


def load_data(path=None):
    """Prebere attrition.csv: vrne (X, feature_names)."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "attrition.csv")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    X = np.array(rows[1:], dtype=float)
    return X, header


def mann_whitney_auc(group, other):
    """AUC = P(vrednost gruče < vrednost zunaj gruče), vezi pol."""
    raise NotImplementedError


if __name__ == "__main__":
    X, names = load_data()
    C, labels = kmeans(X, 3, seed=0)
    g = X[labels == 0, 0]
    o = X[labels != 0, 0]
    print(f"{names[0]}: AUC={mann_whitney_auc(g, o):.3f}")
