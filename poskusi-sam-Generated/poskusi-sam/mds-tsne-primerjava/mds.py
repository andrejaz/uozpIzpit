import numpy as np


def mds(distances, dim=2, lr=0.01, epochs=2000, seed=0):
    """Pomožni MDS za raziskovanje (na voljo za primerjavo)."""
    items = sorted(set(i for pair in distances for i in pair))
    idx = {it: k for k, it in enumerate(items)}
    rng = np.random.default_rng(seed)
    Y = rng.normal(0, 1.0, (len(items), dim))
    pairs = [(idx[a], idx[b], t) for (a, b), t in distances.items()]
    for _ in range(epochs):
        grad = np.zeros_like(Y)
        for i, j, t in pairs:
            diff = Y[i] - Y[j]
            dist = float(np.sqrt(np.sum(diff ** 2))) + 1e-12
            coeff = 2.0 * (dist - t) / dist
            grad[i] += coeff * diff
            grad[j] -= coeff * diff
        Y = Y - lr * grad
    return {it: Y[idx[it]] for it in items}
