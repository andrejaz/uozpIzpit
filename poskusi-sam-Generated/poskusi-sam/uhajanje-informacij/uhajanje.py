import csv
import os

import numpy as np


def load_data(path=None):
    """Prebere body-fat-brozek.csv in vrne (X, y, feature_names)."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "body-fat-brozek.csv")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    data = np.array(rows[1:], dtype=float)
    y = data[:, 0]
    X = data[:, 1:]
    return X, y, header[1:]


def fit_least_squares(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    Xb = np.hstack([X, np.ones((len(X), 1))])
    coef, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return coef[:-1], coef[-1]


def standardize_fit(X_train):
    """Vrne (mean, std) po stolpcih, izračunana samo iz učnih podatkov."""
    raise NotImplementedError


def standardize_apply(X, mean, std):
    """Vrne (X - mean) / std."""
    raise NotImplementedError


def r2_score(y_true, y_pred):
    """Vrne R² = 1 - SS_res / SS_tot."""
    raise NotImplementedError


if __name__ == "__main__":
    X, y, names = load_data()
    print("oblika X:", X.shape, "atributi:", len(names))
    mean, std = standardize_fit(X[:10])
    print("povprecja (ucni):", np.round(mean[:3], 2))
