"""Ustvari sintetično množico cluster-data.csv (zvezni atributi + regija).

Prvih 12 primerov ima nadpovprečno visok f0 in f1 (skrita "gruča").
"""
import csv
import os

import numpy as np


def generate(seed=0, n=80):
    rng = np.random.default_rng(seed)
    f0 = rng.normal(0, 1, n)
    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(0, 1, n)
    f3 = rng.normal(0, 1, n)
    f4 = rng.normal(0, 1, n)
    f0[:12] += 3.0
    f1[:12] += 2.0
    region = rng.choice(["A", "B", "C"], n)
    names = ["f0", "f1", "f2", "f3", "f4", "region"]
    out = os.path.join(os.path.dirname(__file__), "cluster-data.csv")
    with open(out, "w", newline="", encoding="utf-8") as fp:
        wr = csv.writer(fp)
        wr.writerow(names)
        for i in range(n):
            wr.writerow([round(float(f0[i]), 3), round(float(f1[i]), 3),
                         round(float(f2[i]), 3), round(float(f3[i]), 3),
                         round(float(f4[i]), 3), region[i]])
    print(f"zapisano: {out}  ({n} primerov)")


if __name__ == "__main__":
    generate()
