import numpy as np


def fit_least_squares(X, y):
    """Navadna linearna regresija (metoda najmanjših kvadratov). Vrne (w, b)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    Xb = np.hstack([X, np.ones((len(X), 1))])
    coef, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return coef[:-1], coef[-1]


def r2_score(y_true, y_pred):
    """Vrne R² = 1 - SS_res / SS_tot."""
    raise NotImplementedError


def polynomial_features(x, degree):
    """Za vektor x vrne matriko stolpcev [x, x^2, ..., x^degree]."""
    raise NotImplementedError


def train_test_split(X, y, test_ratio, seed=0):
    """Naključno razdeli podatke. Vrne (X_train, X_test, y_train, y_test)."""
    raise NotImplementedError


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 20)
    y = np.sin(3 * x) + rng.normal(0, 0.1, 20)
    for deg in [1, 3, 9]:
        Xtr, Xte, ytr, yte = train_test_split(polynomial_features(x, deg), y, 0.3)
        w, b = fit_least_squares(Xtr, ytr)
        pred = Xte @ w + b
        print(f"stopnja={deg}  testni R2={r2_score(yte, pred):.3f}")
