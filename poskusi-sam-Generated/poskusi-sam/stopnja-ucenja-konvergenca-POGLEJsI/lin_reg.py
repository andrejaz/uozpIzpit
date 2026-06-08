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

def grad_w(x,y):
    return 2*np.mean((p-y)*x)

def grad_b(x,y):
    return 2*np.mean(p-y)


def train(xs, ys, lr, n_epochs):
    """Univariatna linearna regresija z gradientnim sestopom. Vrne (w, b).
    Začni pri w = 0, b = 0."""
    w = 0.0
    b = 0.0
    for _ in range(n_epochs):
        # 1. Napoved (p)
        p = predict(xs, w, b)
        
        # 2. Izračun gradientov
        # dL/dw = 2 * mean((p - y) * x)
        dw = 2 * np.mean((p - ys) * xs)
        # dL/db = 2 * mean(p - y)
        db = 2 * np.mean(p - ys)
        
        # 3. Posodobitev uteži
        w -= lr * dw
        b -= lr * db
    
    return w,b



if __name__ == "__main__":
    xs, ys = make_data()
    for lr in [0.001, 0.01, 0.1, 1.0, 10.0]:
        try:
            w, b = train(xs, ys, lr, n_epochs=200)
            print(f"lr={lr:<6} w={w:.3f} b={b:.3f} loss={mse_loss(xs, ys, w, b):.3f}")
        except OverflowError:
            print(f"lr={lr:<6} OverflowError (divergenca)")
