"""Ustvari sintetično množico hdi.csv (države x socioekonomske značilke).

Države izhajajo iz treh "profilov", kar ustvari naravne skupine.
"""
import csv
import os

import numpy as np


def generate(seed=0, n_per=15, d=10):
    rng = np.random.default_rng(seed)
    profiles = [rng.normal(0, 1, d) * 2 for _ in range(3)]
    rows = []
    countries = []
    idx = 1
    for p, prof in enumerate(profiles):
        for _ in range(n_per):
            rows.append(prof + rng.normal(0, 0.5, d))
            countries.append(f"Drzava_{idx:02d}")
            idx += 1
    X = np.array(rows)
    names = [f"znacilka_{j}" for j in range(d)]
    out = os.path.join(os.path.dirname(__file__), "hdi.csv")
    with open(out, "w", newline="", encoding="utf-8") as fp:
        wr = csv.writer(fp)
        wr.writerow(["country"] + names)
        for c, r in zip(countries, X):
            wr.writerow([c] + [round(float(v), 3) for v in r])
    print(f"zapisano: {out}  ({len(countries)} drzav, {d} znacilk)")


if __name__ == "__main__":
    generate()
