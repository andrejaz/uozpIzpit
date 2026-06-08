import numpy as np


def make_data(n=300, seed=0):
    """5 atributov: dva para koreliranih (iz dveh skritih spremenljivk) in
    en šumni atribut."""
    rng = np.random.default_rng(seed)
    z1 = rng.normal(0, 1, n)
    z2 = rng.normal(0, 1, n)
    X = np.column_stack([
        z1 + rng.normal(0, 0.1, n),
        z1 + rng.normal(0, 0.1, n),
        z2 + rng.normal(0, 0.1, n),
        z2 + rng.normal(0, 0.1, n),
        rng.normal(0, 1, n),
    ])
    return X


def standardize_fit(X):
    X = np.asarray(X, dtype=float)
    return X.mean(axis=0), X.std(axis=0)


def standardize_apply(X, mean, std):
    return (np.asarray(X, dtype=float) - mean) / std


def pca_fit(X_train_std):
    """Vrne matriko komponent (vrstice so glavne komponente, padajoče)."""
    raise NotImplementedError


def explained_on_test(X_test_std, components, k):
    """Delež variance testnih podatkov, ki ga pojasni prvih k komponent."""
    raise NotImplementedError


if __name__ == "__main__":
    X = make_data()
    mean, std = standardize_fit(X[:50])
    comps = pca_fit(standardize_apply(X[:50], mean, std))
    Xte = standardize_apply(X[50:], mean, std)
    for k in [1, 2, 3, 5]:
        print(f"k={k}  pojasnjena varianca na testu={explained_on_test(Xte, comps, k):.3f}")
