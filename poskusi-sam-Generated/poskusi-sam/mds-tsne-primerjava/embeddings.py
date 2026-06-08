import numpy as np


def make_groups_highdim(n_per, dim, sep, seed=0):
    """Tri Gaussove skupine v dim-razsežnem prostoru, ločene z razmikom sep.
    Vrne (X, labels)."""
    raise NotImplementedError


def silhouette(X, labels):
    """Povprečna silhueta razbitja (ločljivost skupin)."""
    raise NotImplementedError


if __name__ == "__main__":
    X, labels = make_groups_highdim(n_per=30, dim=10, sep=6.0, seed=0)
    print("oblika:", X.shape, "silhueta:", round(silhouette(X, labels), 3))
