import numpy as np


def pca_gradient(X, steps=800, lr=0.05, seed=0):
    """Prva glavna komponenta z gradientnim vzponom (za primerjavo)."""
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0)
    n = len(Xc)
    cov = (Xc.T @ Xc) / n
    rng = np.random.default_rng(seed)
    u = rng.normal(size=Xc.shape[1])
    u = u / np.linalg.norm(u)
    for _ in range(steps):
        g = 2 * cov @ u
        u = u + lr * g
        u = u / np.linalg.norm(u)
    return u


def analytic_pca(X):
    """Vrne (components, eigenvalues, ratios); vrstice components so enotni
    lastni vektorji kovariančne matrike, urejeni po padajoči lastni vrednosti."""
    raise NotImplementedError


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (300, 3)) @ np.array([[3.0, 0, 0], [0, 1.0, 0], [0, 0, 0.5]])
    comps, vals, ratios = analytic_pca(X)
    print("lastne vrednosti:", np.round(vals, 3))
    print("delezi:", np.round(ratios, 3))
    u = pca_gradient(X)
    print("kosinus(prva komp., gradient):",
          round(abs(float(comps[0] @ u)), 4))
