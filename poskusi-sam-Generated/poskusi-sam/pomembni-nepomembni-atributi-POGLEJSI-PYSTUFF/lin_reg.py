import numpy as np


def predict(X, w, b):
    return np.asarray(X, dtype=float) @ w + b


def make_data(seed=42, n=1000, n_noise=5, noise=1.0):
    """Prvi 3 atributi so informativni (y = 2x0 + 3x1 - x2 + 1 + šum),
    naslednjih n_noise atributov pa je naključnih in nepovezanih z y."""
    rng = np.random.default_rng(seed)
    X_inf = rng.standard_normal((n, 3))
    y = 2 * X_inf[:, 0] + 3 * X_inf[:, 1] - X_inf[:, 2] + 1 + rng.normal(0, noise, n)
    X_noise = rng.standard_normal((n, n_noise))
    X = np.hstack([X_inf, X_noise])
    return X, y


def train_multi(X, y, lr, n_epochs):
    """Multivariatna linearna regresija z gradientnim sestopom. Vrne (w, b)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(n_epochs):
        p = X @ w + b
        gw = 2.0 / n * (X.T @ (p - y))
        gb = 2.0 / n * np.sum(p - y)
        w -= lr * gw
        b -= lr * gb
    return w, b


def rank_features(w):
    """Vrne indekse atributov, urejene padajoče po |w|."""
    w_abs = np.abs(w)
    # np.argsort vrne indekse, ki bi sortirali w_abs od najmanjšega do največjega
    indeksi = np.argsort(w_abs)
    # Obrnemo vrstni red (::-1), da dobimo največje na začetku
    return indeksi[::-1]

    


if __name__ == "__main__":
    X, y = make_data(noise=1.0)
    w, b = train_multi(X, y, lr=0.05, n_epochs=2000)
    order = rank_features(w)
    print("razvrstitev (od najpomembnejsega):", order)
