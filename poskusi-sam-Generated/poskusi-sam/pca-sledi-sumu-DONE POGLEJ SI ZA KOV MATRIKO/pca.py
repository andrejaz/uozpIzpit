import numpy as np


def first_principal_component(X):
    """Enotni smerni vektor prve glavne komponente."""
    #centriranje podatkov
    X_centered = X - np.mean(X, axis=0)

    #kovariancna matrika 
    C = np.cov(X_centered, rowvar=False)

    #lastne vrednosti in lastni vektroji
    w, v = np.linalg.eigh(C)

    return v[:, -1]




def make_two_groups(noise=0.0, n=100, sep=4.0, seed=0):
    """Dve skupini, ločeni v vodoravni smeri (x = -sep in x = +sep),
    z navpičnim (y) šumom standardnega odklona 'noise'."""
    rng = np.random.default_rng(seed)
    g1 = np.column_stack([-sep + rng.normal(0, 0.3, n), rng.normal(0, noise, n)])
    g2 = np.column_stack([sep + rng.normal(0, 0.3, n), rng.normal(0, noise, n)])
    return np.vstack([g1, g2])


if __name__ == "__main__":
    for noise in [0.1, 1.0, 3.0, 6.0, 10.0]:
        v = first_principal_component(make_two_groups(noise=noise, seed=0))
        print(f"noise={noise:<5} smer=({v[0]:+.2f}, {v[1]:+.2f})")
