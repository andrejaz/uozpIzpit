import numpy as np


def gaussian_affinities(D, sigma):
    """Pogojne verjetnosti p_{j|i} iz matrike razdalj D (vrstice normirane)."""
    raise NotImplementedError


def make_clusters(n_per, sep, seed=0):
    """Tri 2D skupine, ločene z razmikom sep. Vrne (X, labels)."""
    raise NotImplementedError


if __name__ == "__main__":
    X, labels = make_clusters(20, 10.0)
    print("oblika:", X.shape)
