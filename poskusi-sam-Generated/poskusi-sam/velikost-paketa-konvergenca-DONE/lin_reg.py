import random

import numpy as np


def predict(xs, w, b):
    return w * np.asarray(xs, dtype=float) + b


def mse_loss(xs, ys, w, b):
    p = predict(xs, w, b)
    return float(np.mean((p - np.asarray(ys, dtype=float)) ** 2))


def make_data(seed=42, n=1000, noise=5.0):
    """Univariatni podatki: y = 2x + 1 + šum."""
    random.seed(seed)
    xs = [random.uniform(-10, 10) for _ in range(n)]
    ys = [2 * x + 1 + random.gauss(0, noise) for x in xs]
    return np.array(xs), np.array(ys)


def train_batches(xs, ys, lr, n_epochs, batch_size, seed=0):
    """Paketni (mini-batch) gradientni sestop. Vrne (w, b)."""
    rng = np.random.default_rng(seed)
    w = 0.0
    b = 0.0
    for i in range(n_epochs):
        indices = rng.choice(len(xs), size=batch_size, replace=False)
        xs_batch = xs[indices]
        ys_batch = ys[indices]

        p = predict(xs_batch, w, b)

        dw = 2 * np.mean((p - ys_batch) * xs_batch)
        db = 2 * np.mean(p - ys_batch)

        w -= lr * dw
        b -= lr * db
    
    return (w,b)



if __name__ == "__main__":
    xs, ys = make_data()
    for bs in [1, 5, 10, 50, 100, 1000]:
        w, b = train_batches(xs, ys, 0.01, n_epochs=500, batch_size=bs, seed=0)
        print(f"batch={bs:<5} w={w:.3f} b={b:.3f} loss={mse_loss(xs, ys, w, b):.3f}")
