"""Ustvari sintetično množico attrition.csv (numerične značilke zaposlenih)."""
import csv
import os

import numpy as np


def generate(seed=0, n=300):
    rng = np.random.default_rng(seed)
    age = rng.normal(37, 9, n).clip(20, 60)
    income = rng.normal(6500, 2500, n).clip(1500, 20000)
    satisfaction = rng.integers(1, 5, n).astype(float)
    overtime = rng.normal(5, 4, n).clip(0, 25)
    years = rng.normal(7, 5, n).clip(0, 35)
    distance = rng.normal(9, 7, n).clip(1, 30)
    names = ["Age", "MonthlyIncome", "JobSatisfaction",
             "OverTimeHours", "YearsAtCompany", "DistanceFromHome"]
    cols = np.column_stack([age, income, satisfaction, overtime, years, distance])
    out = os.path.join(os.path.dirname(__file__), "attrition.csv")
    with open(out, "w", newline="", encoding="utf-8") as fp:
        wr = csv.writer(fp)
        wr.writerow(names)
        for r in cols:
            wr.writerow([round(float(v), 2) for v in r])
    print(f"zapisano: {out}  ({n} zaposlenih)")


if __name__ == "__main__":
    generate()
