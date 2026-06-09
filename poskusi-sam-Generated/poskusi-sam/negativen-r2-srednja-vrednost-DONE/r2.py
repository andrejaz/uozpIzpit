import numpy as np


def r2_score(y_true, y_pred):
    """Vrne R² = 1 - SS_res / SS_tot."""
    y_mean = np.mean(y_true)
    SS_res = np.sum((y_true-y_pred)**2)
    SS_tot = np.sum((y_true-y_mean)**2)
    return 1 - (SS_res/SS_tot)


def baseline_predict(y_train, n_test):
    """Vrne polje dolžine n_test s povprečjem y_train."""
    r = list()
    mean = np.mean(y_train)
    for i in range(n_test):
        r.append(mean)
    return r



if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 100)
    y_train, y_test = y[:50], y[50:]
    pred = baseline_predict(y_train, len(y_test))
    print("R2 (srednja vrednost):", r2_score(y_test, pred))
