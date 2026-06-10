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


def silhouette_score(X, labels):
    """Povprečna silhueta razbitja."""
    raise NotImplementedError


def best_k(X, ks, seed=0):
    """Vrne k iz ks z največjo silhueto (z metodo voditeljev)."""
    


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, 2))  # brez strukture
    print("najboljsi k:", best_k(X, [2, 3, 4, 5, 6, 7, 8]))
