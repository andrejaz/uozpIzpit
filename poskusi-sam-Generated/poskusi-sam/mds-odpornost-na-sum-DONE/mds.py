import numpy as np


DISTANCES = {
    ("Novo Mesto", "Maribor"): 170,
    ("Novo Mesto", "Celje"): 83,
    ("Novo Mesto", "Koper"): 169,
    ("Novo Mesto", "Kranj"): 99,
    ("Novo Mesto", "Ljubljana"): 72,
    ("Novo Mesto", "Postojna"): 116,
    ("Maribor", "Celje"): 55,
    ("Maribor", "Koper"): 232,
    ("Maribor", "Kranj"): 156,
    ("Maribor", "Ljubljana"): 128,
    ("Maribor", "Postojna"): 178,
    ("Celje", "Koper"): 183,
    ("Celje", "Kranj"): 105,
    ("Celje", "Ljubljana"): 77,
    ("Celje", "Postojna"): 130,
    ("Koper", "Kranj"): 128,
    ("Koper", "Ljubljana"): 107,
    ("Koper", "Postojna"): 58,
    ("Kranj", "Ljubljana"): 30,
    ("Kranj", "Postojna"): 77,
    ("Ljubljana", "Postojna"): 53,
}


def stress(positions, distances):
    s = 0.0
    for (a, b), target in distances.items():
        diff = np.asarray(positions[a], float) - np.asarray(positions[b], float)
        s += (float(np.sqrt(np.sum(diff ** 2))) - target) ** 2
    return float(s)


def mds(distances, dim=2, lr=0.01, epochs=2000, seed=0):
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


def perturb_distance(distances, pair, delta):
    """Vrne kopijo slovarja s spremenjeno razdaljo za pair (in simetrični par)."""
    dist_new = distances.copy()
    if pair in dist_new: 
        dist_new[pair] += delta
    else:
        reverse = (pair[1], pair[0])
        dist_new[reverse] += delta

    return dist_new


def total_movement(pos_a, pos_b):
    """Vsota evklidskih razdalj med položaji istih mest v dveh vložitvah."""
    # return np.linalg.norm(pos_a+pos_b)
    dist = 0
    for mesto in pos_a:
        v1 = pos_a[mesto]
        v2 = pos_b[mesto]
        dist += np.linalg.norm(v1-v2)

    return dist


if __name__ == "__main__":
    d2 = perturb_distance(DISTANCES, ("Ljubljana", "Koper"), 100)
    a = mds(DISTANCES)
    b = mds(d2)
    print("skupni premik:", round(total_movement(a, b), 2))
