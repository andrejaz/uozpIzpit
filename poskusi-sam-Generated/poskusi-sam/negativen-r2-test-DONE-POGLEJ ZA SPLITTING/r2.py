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
    n = len(y_true)
    y_mean = np.mean(y_true)
    SS_res = np.sum((y_true - y_pred)**2)
    SS_tot = np.sum((y_true-y_mean)**2)

    return 1 - (SS_res/SS_tot)



def polynomial_features(x, degree):
    """Za vektor x vrne matriko stolpcev [x, x^2, ..., x^degree]."""
    n = len(x)
    m = []
    
    for i in range(n):
        vrstica = []
        for d in range(1, degree + 1):
            vrstica.append(x[i]**d)
        m.append(vrstica)
    
    # Pretvorimo v numpy matriko, da bo delovalo z matrikami
    return np.array(m)


def train_test_split(X, y, test_ratio, seed=0):
    """Naključno razdeli podatke. Vrne (X_train, X_test, y_train, y_test)."""
    
    X = np.asarray(X)
    y = np.asarray(y)
    
    n = len(X)
    indices = np.arange(n, dtype=int) # Zagotovi, da so indeksi int
    
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    
    n_test = int(n * test_ratio)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # 2. Uporaba fancy indexinga, ki sedaj MORA delovati, ker sta X in y numpy polji
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]



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
