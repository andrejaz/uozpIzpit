import random

import numpy as np


def predict(xs, w, b):
    return w * np.asarray(xs, dtype=float) + b


def mse_loss(xs, ys, w, b):
    p = predict(xs, w, b)
    return float(np.mean((p - np.asarray(ys, dtype=float)) ** 2))


def make_data(seed=42, n=5, noise=4.0):
    """Univariatni podatki: y = 2x - 1 + šum."""
    random.seed(seed)
    xs = [random.uniform(-5, 5) for _ in range(n)]
    ys = [2 * x - 1 + random.gauss(0, noise) for x in xs]
    return np.array(xs), np.array(ys)


def train(xs, ys, lr, n_epochs):
    """Univariatna linearna regresija z gradientnim sestopom (w=0, b=0)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    w, b = 0.0, 0.0
    n = len(xs)
    for _ in range(n_epochs):
        p = w * xs + b
        gw = 2.0 / n * np.sum((p - ys) * xs)
        gb = 2.0 / n * np.sum(p - ys)
        w -= lr * gw
        b -= lr * gb
    return w, b


def diverged(xs, ys, lr, n_epochs):
    """Vrne True, če učenje pri dani stopnji učenja podivja."""
    w_train, b_train = train(xs, ys, lr, n_epochs)
    return not (np.isfinite(w_train) and np.isfinite(b_train))


def find_critical_lr(xs, ys, n_epochs=200, start=1e-3, factor=2.0):
    """Z večanjem in bisekcijo poišče največjo stabilno stopnjo učenja."""
    stabilna = start
    while not diverged(xs, ys, stabilna * factor, n_epochs):
        stabilna *= factor
    
    nestabilna = stabilna * factor
    
    # 2. Bisekcija
    # Iščemo največjo stabilno vrednost, zato bomo interval ožili
    for _ in range(20): # 20 iteracij zadostuje za zelo visoko natančnost
        sredina = (stabilna + nestabilna) / 2
        if diverged(xs, ys, sredina, n_epochs):
            nestabilna = sredina # Sredina je še vedno preveč, premakni mejo
        else:
            stabilna = sredina   # Sredina je stabilna, to je zdaj spodnja meja
            
    return stabilna





if __name__ == "__main__":
    xs, ys = make_data()
    print("kriticna stopnja ucenja:", find_critical_lr(xs, ys))
