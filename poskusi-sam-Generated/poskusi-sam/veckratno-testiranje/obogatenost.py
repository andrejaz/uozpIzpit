import math


def enrichment(all_items, selected_items, group):
    """Vrne slovar z N, K, n, k, expected, fold, p_value (hipergeometrijski
    P(X >= k))."""
    raise NotImplementedError


def count_significant(all_items, selected_items, groups, alpha=0.05):
    """Vrne število skupin s p_value < alpha."""
    raise NotImplementedError


if __name__ == "__main__":
    all_items = set(range(12))
    selected = {0, 1, 2, 3, 11}
    print(enrichment(all_items, selected, set(range(5))))
