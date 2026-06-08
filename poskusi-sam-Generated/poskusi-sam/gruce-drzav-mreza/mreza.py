import csv
import os

import numpy as np


def load_data(path=None):
    """Prebere hdi.csv: vrne (countries, X, feature_names)."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "hdi.csv")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    countries = [r[0] for r in rows[1:]]
    X = np.array([[float(v) for v in r[1:]] for r in rows[1:]])
    return countries, X, header[1:]


def cosine_distance(a, b):
    """Vrne 1 - kosinusna podobnost med vektorjema a in b."""
    raise NotImplementedError


def build_graph(X, threshold):
    """Vrne slovar sosednosti {i: množica sosedov} (cosine_distance <= threshold)."""
    raise NotImplementedError


def label_propagation(adj, seed=0, max_iter=100):
    """Z razširjanjem oznak vrne slovar {vozlišče: oznaka skupnosti}."""
    raise NotImplementedError


if __name__ == "__main__":
    countries, X, names = load_data()
    Xs = (X - X.mean(axis=0)) / X.std(axis=0)
    adj = build_graph(Xs, threshold=0.3)
    labels = label_propagation(adj, seed=0)
    print("st. skupnosti:", len(set(labels.values())))
