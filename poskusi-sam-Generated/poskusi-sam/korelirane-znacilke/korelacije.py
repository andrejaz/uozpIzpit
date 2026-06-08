import numpy as np

def fit(X, y, reg=None, reg_strength=0.0, lr=0.1, n_epochs=3000, seed=0):
    """Linearna regresija z gradientnim sestopom in (po želji) L1/L2
    regularizacijo. Atribute standardizira, zato so uteži primerljive.
    Vrne (w, b)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, d)
    b = float(y.mean())
    for _ in range(n_epochs):
        err = Xs @ w + b - y
        gw = 2.0 / n * (Xs.T @ err)
        gb = 2.0 / n * float(np.sum(err))
        if reg == "l2":
            gw = gw + 2.0 * reg_strength * w
        elif reg == "l1":
            gw = gw + reg_strength * np.sign(w)
        w = w - lr * gw
        b = b - lr * gb
        if reg == "l1":
            thr = lr * reg_strength
            w[np.abs(w) < thr] = 0.0
    return w, b


def make_correlated_data(seed=0, n=200):
    """Cilj je odvisen od a, b, c; a2 in a3 sta močno korelirana z a."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    c = rng.normal(0, 1, n)
    a2 = a + rng.normal(0, 0.03, n)
    a3 = a + rng.normal(0, 0.03, n)
    X = np.column_stack([a, b, c, a2, a3])
    y = 2 * a - 1.5 * b + c + rng.normal(0, 0.3, n)
    return X, y


def selected_features(w, tol=1e-8):
    """Vrne množico indeksov atributov z neničelno utežjo (|w| > tol)."""
    raise NotImplementedError


if __name__ == "__main__":
    X, y = make_correlated_data()
    for seed in range(5):
        w, b = fit(X, y, reg="l1", reg_strength=0.2, seed=seed)
        print(sorted(selected_features(w)))
