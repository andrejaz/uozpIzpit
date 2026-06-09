import numpy as np


def gaussian_affinities(D, sigma):
    """Pogojne verjetnosti p_{j|i} iz matrike razdalj D (vrstice normirane)."""
    w = np.exp(-(D**2) / (2 * sigma**2))
    np.fill_diagonal(w,0)
    row_sums = np.sum(w, axis=1, keepdims=True)
    # da delis vrednosti v vrstici da normaliziras 
    p = np.divide(w, row_sums, out=np.zeros_like(w), where=row_sums != 0)

    return p


def make_clusters(n_per, sep, seed=0):
    """Tri 2D skupine, ločene z razmikom sep. Vrne (X, labels)."""
    rng = np.random.default_rng(seed)
    centers = [np.array([0,0]), np.array([sep, 0]), np.array([0, sep])]
    
    X = []
    labels = []
    for i, center in enumerate(centers):
        cluster = center + rng.standard_normal((n_per, 2))
        X.append(cluster)
        labels.append(np.full(n_per, i))

    X = np.vstack(X)
    labels = np.concatenate(labels)

    return X, labels


if __name__ == "__main__":
    X, labels = make_clusters(20, 10.0)
    print("oblika:", X.shape)
