import numpy as np


def predict(X, w, b):
    return np.asarray(X, dtype=float) @ w + b


def mse_loss(X, y, w, b):
    return float(np.mean((predict(X, w, b) - np.asarray(y, dtype=float)) ** 2))


def make_data(seed=42, n=1000, noise=0.0):
    """Multivariatni podatki: y = 2x0 + 3x1 - x2 + 1 + šum.
    Atributi so vzorčeni iz standardne normalne porazdelitve."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 1 + rng.normal(0, noise, n)
    return X, y


def train_multi(X, y, lr, n_epochs):
    """Multivariatna linearna regresija z gradientnim sestopom. Vrne (w, b).
    Začni pri w = ničle, b = 0."""
    w = 0.0
    b = 0.0
    
    for i in range(n_epochs):
        p = predict(X, w,b)
        dw = 2 * np.mean((p - y) * X)
        db = 2 * np.mean(p - y)

        w -= lr * dw
        b -= lr * db


    return ()


if __name__ == "__main__":
    for noise in [0.0, 1.0, 5.0, 20.0]:
        X, y = make_data(noise=noise)
        w, b = train_multi(X, y, lr=0.05, n_epochs=3000)
        print(f"noise={noise:<5} w={np.round(w, 3)} b={b:.3f}")
