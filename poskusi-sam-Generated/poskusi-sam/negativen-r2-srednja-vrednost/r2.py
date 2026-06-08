import numpy as np


def r2_score(y_true, y_pred):
    """Vrne R² = 1 - SS_res / SS_tot."""
    raise NotImplementedError


def baseline_predict(y_train, n_test):
    """Vrne polje dolžine n_test s povprečjem y_train."""
    raise NotImplementedError


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 100)
    y_train, y_test = y[:50], y[50:]
    pred = baseline_predict(y_train, len(y_test))
    print("R2 (srednja vrednost):", r2_score(y_test, pred))
