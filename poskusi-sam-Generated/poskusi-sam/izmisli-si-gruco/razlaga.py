import csv
import os

import numpy as np


def load_data(path=None):
    """Prebere cluster-data.csv: vrne (X, feature_names, region)."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "cluster-data.csv")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    region = [r[-1] for r in rows[1:]]
    X = np.array([[float(v) for v in r[:-1]] for r in rows[1:]])
    return X, header[:-1], region


def t_statistic(group_vals, other_vals):
    """(Welchova) t-statistika med dvema skupinama vrednosti."""
    raise NotImplementedError


def rank_features_ttest(X, mask):
    """Indeksi atributov, urejeni padajoče po |t-statistiki|."""
    raise NotImplementedError


if __name__ == "__main__":
    X, names, region = load_data()
    mask = np.zeros(len(X), dtype=bool)
    mask[:12] = True  # izmišljena gruča: prvih 12 primerov
    order = rank_features_ttest(X, mask)
    for i in order:
        print(f"{names[i]:<12} t={t_statistic(X[mask, i], X[~mask, i]):+.2f}")
