"""Ustvari sintetično različico podatkovne množice body-fat-brozek.csv.

Cilj (delež telesne maščobe) je najbolj odvisen od obsega trebuha (abdomen),
kar posnema strukturo izvirne množice iz poglavja o regularizaciji.
"""
import csv
import os

import numpy as np


def generate(seed=0, n=252):
    rng = np.random.default_rng(seed)
    age = rng.normal(45, 12, n).clip(20, 80)
    height = rng.normal(178, 7, n)
    weight = rng.normal(80, 12, n)
    abdomen = 70 + 0.6 * (weight - 80) + 0.2 * (age - 45) + rng.normal(0, 5, n)
    hip = 95 + 0.4 * (weight - 80) + rng.normal(0, 4, n)
    chest = 95 + 0.5 * (weight - 80) + rng.normal(0, 4, n)
    neck = 37 + 0.10 * (weight - 80) + rng.normal(0, 1.5, n)
    thigh = 58 + 0.30 * (weight - 80) + rng.normal(0, 3, n)
    knee = 38 + 0.10 * (weight - 80) + rng.normal(0, 1.5, n)
    ankle = 23 + 0.05 * (weight - 80) + rng.normal(0, 1.0, n)
    biceps = 32 + 0.15 * (weight - 80) + rng.normal(0, 2.0, n)
    forearm = 28 + 0.10 * (weight - 80) + rng.normal(0, 1.5, n)
    wrist = 18 + 0.05 * (weight - 80) + rng.normal(0, 0.8, n)
    adiposity = weight / ((height / 100) ** 2)
    bodyfat = (-40 + 0.9 * abdomen - 0.2 * weight + 0.1 * age
               + rng.normal(0, 3, n)).clip(2, 45)

    names = ["body fat brozek", "age", "weight", "height", "neck", "chest",
             "abdomen", "hip", "thigh", "knee", "ankle", "biceps", "forearm",
             "wrist", "adiposity"]
    cols = [bodyfat, age, weight, height, neck, chest, abdomen, hip, thigh,
            knee, ankle, biceps, forearm, wrist, adiposity]
    rows = np.column_stack(cols)
    out = os.path.join(os.path.dirname(__file__), "body-fat-brozek.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(names)
        for r in rows:
            writer.writerow([round(float(v), 2) for v in r])
    print(f"zapisano: {out}  ({n} primerov, {len(names) - 1} atributov)")


if __name__ == "__main__":
    generate()
