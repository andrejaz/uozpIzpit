import numpy as np


def assign(X, leaders):
    """Vrne indeks najbližjega voditelja za vsak primer."""
    raise NotImplementedError


def train_leaders(X, k, lr=0.5, steps=300, seed=0):
    """Gradientni sestop nad koordinatami voditeljev. Vrne matriko (k, d)."""
    raise NotImplementedError


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal([0, 0], 0.5, (50, 2)),
                   rng.normal([10, 10], 0.5, (50, 2))])
    leaders = train_leaders(X, 2, seed=0)
    print("voditelji:\n", np.round(leaders, 2))
