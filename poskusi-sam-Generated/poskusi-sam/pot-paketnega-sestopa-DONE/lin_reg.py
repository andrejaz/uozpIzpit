import random

import numpy as np


def predict(xs, w, b):
    return w * np.asarray(xs, dtype=float) + b


def mse_loss(xs, ys, w, b):
    p = predict(xs, w, b)
    return float(np.mean((p - np.asarray(ys, dtype=float)) ** 2))


def make_data(seed=42, n=1000, noise=5.0):
    random.seed(seed)
    xs = [random.uniform(-10, 10) for _ in range(n)]
    ys = [2 * x + 1 + random.gauss(0, noise) for x in xs]
    return np.array(xs), np.array(ys)


def train_batches_with_history(xs, ys, lr, n_epochs, batch_size, seed=0):
    """Vrne seznam parov (w, b) dolžine n_epochs + 1; prvi je (0.0, 0.0)."""
    w = 0.0
    b = 0.0
    rez = []
    rez.append((w,b))
    
    for i in range(n_epochs):
        p = predict(xs, w,b)
        dw = 2 * np.mean((p - ys) * xs)
        db = 2 * np.mean(p - ys)

        w -= lr * dw
        b -= lr * db

        rez.append((w,b))

    return rez


if __name__ == "__main__":
    xs, ys = make_data()
    path = train_batches_with_history(xs, ys, 0.01, 200, batch_size=10)
    print("dolzina:", len(path), "konec:", path[-1])
