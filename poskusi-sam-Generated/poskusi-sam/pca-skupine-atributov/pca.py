import numpy as np


def make_grouped_data(n=300, noise=0.1, seed=0):
    """9 atributov: prvi 3 iz skrite spremenljivke h1, naslednji 3 iz h2,
    zadnji 3 iz h3; vsak atribut ima dodan šum. Podatki so standardizirani."""
    rng = np.random.default_rng(seed)
    hidden = [rng.normal(0, 1, n) for _ in range(3)]
    cols = []
    for h in hidden:
        for _ in range(3):
            cols.append(h + rng.normal(0, noise, n))
    X = np.column_stack(cols)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X


def top_components(X, k):
    """Vrne matriko (k, d) z utežmi prvih k glavnih komponent."""
    raise NotImplementedError


def explained_variance_ratios(X):
    """Vrne deleže razložene variance vseh komponent (padajoče)."""
    raise NotImplementedError


if __name__ == "__main__":
    X = make_grouped_data(noise=0.1)
    ratios = explained_variance_ratios(X)
    print("delezi prvih treh komponent:", np.round(ratios[:3], 3))
    print("skupaj prve tri:", round(float(np.sum(ratios[:3])), 3))
