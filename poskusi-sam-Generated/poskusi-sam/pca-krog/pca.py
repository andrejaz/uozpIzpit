import numpy as np


def first_principal_component(X):
    """Prva glavna komponenta: lastni vektor kovariančne matrike z
    največjo lastno vrednostjo (enotni smerni vektor)."""
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0)
    cov = (Xc.T @ Xc) / len(Xc)
    vals, vecs = np.linalg.eigh(cov)
    return vecs[:, -1]


def make_circle(n=200, seed=0):
    """Točke, enakomerno razporejene po enotski krožnici."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([np.cos(t), np.sin(t)])


def explained_variance_ratio(X, v):
    """Delež razložene variance v smeri v (enotni vektor u = v / ||v||)."""
    raise NotImplementedError


if __name__ == "__main__":
    X = make_circle(500)
    v = first_principal_component(X)
    print("delez razlozene variance prve komponente:",
          round(explained_variance_ratio(X, v), 3))
