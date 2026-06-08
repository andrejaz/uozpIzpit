import numpy as np


def explained_variance_first(X):
    """Delež variance prve glavne komponente
    (največja lastna vrednost / vsota lastnih vrednosti kovariančne matrike)."""
    raise NotImplementedError


def mean_explained_first(n, d, reps, seed=0):
    """Povprečje deležev variance prve komponente čez 'reps' naključnih
    standardiziranih množic (n x d, standardna normalna porazdelitev)."""
    raise NotImplementedError


if __name__ == "__main__":
    for n in [20, 50, 100, 500, 5000]:
        m = mean_explained_first(n, 10, reps=20, seed=0)
        print(f"n={n:<6} povprecni delez prve komponente={m:.3f}  (1/d=0.100)")
