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


def make_data(seed=0, n=200):
    """3 informativni + 3 nepomembni + 3 korelirani (kopije prvega) atributi."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, (n, 3))
    c1 = x1 + rng.normal(0, 0.05, n)
    c2 = x1 + rng.normal(0, 0.05, n)
    c3 = x1 + rng.normal(0, 0.05, n)
    X = np.column_stack([x1, x2, x3, noise, c1, c2, c3])
    y = 3 * x1 - 2 * x2 + x3 + rng.normal(0, 0.5, n)
    return X, y


def regularization_path(X, y, reg, lambdas):
    """Vrne matriko uteži oblike (len(lambdas), n_features)."""
    n_features = X.shape[1] #stevilo atributov 

    path = np.zeros((len(lambdas), n_features)) #inicializiramo matriko z niclami, shranjevali bomo
    #tezo za vsako lambdo. vrstice so lambde stolpci pa utezi posameznih atributov 
    for i in range(len(lambdas)):
        l = lambdas[i]
        w,b = fit(X, y, reg, l)
        path[i,:] = w # utezi shranimo v ustrezno vrstico path

    return path



if __name__ == "__main__":
    X, y = make_data()
    P = regularization_path(X, y, "l1", [0.0, 0.1, 0.5, 1.0, 2.0])
    print("oblika poti:", P.shape)
    print("st. nicelnih utezi po lambda:", (np.abs(P) < 1e-9).sum(axis=1))
