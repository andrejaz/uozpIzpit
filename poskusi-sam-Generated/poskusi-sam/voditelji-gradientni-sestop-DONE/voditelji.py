import numpy as np


def assign(X, leaders):
    """Vrne indeks najbližjega voditelja za vsak primer."""
    min_razdalja = 0
    naj_leader = 0
    best_leaders = []
    for i in range(X.shape[0]):
        x = X[i][0]
        y = X[i][1]
        min_razdalja = 0
        naj_leader = 0
        for j in range(leaders.shape[0]):
            lx = leaders[j][0]
            ly = leaders[j][1]

            razdalja = np.sum((X[i] - leaders[j])**2)
            if min_razdalja > razdalja:
                min_razdalja = razdalja
                naj_leader = j
        
        best_leaders.append(naj_leader)

    return np.array(best_leaders)


def train_leaders(X, k, lr=0.5, steps=300, seed=0):
    """Gradientni sestop nad koordinatami voditeljev. Vrne matriko (k, d)."""
    n, d = X.shape
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=k, replace=False)
    leaders = X[indices].copy()   # (k, d)
    
    for i in range(steps):
        assigned_leaders = assign(X, leaders)

        for j in range(k):
            points = X[assigned_leaders == i]
            mean = points.mean(axis=0)
            n_i = len(points)
            gradient = (2.0 * n_i / n) * (leaders[i] - mean)
            # Korak gradientnega sestopa
            leaders[i] = leaders[i] - lr * gradient

    return leaders


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal([0, 0], 0.5, (50, 2)),
                   rng.normal([10, 10], 0.5, (50, 2))])
    leaders = train_leaders(X, 2, seed=0)
    print("voditelji:\n", np.round(leaders, 2))
