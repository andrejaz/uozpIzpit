import math

def nCr(n, r):
    """Pomožna funkcija za binomski koeficient."""
    if r < 0 or r > n:
        return 0
    return math.comb(n, r)


def enrichment(all_items, selected_items, group):
    """Vrne slovar z N, K, n, k, expected, fold, p_value (hipergeometrijski
    P(X >= k))."""
    N = len(all_items)
    K = 0
    n = 0
    k = 0
    for item in selected_items:
        if item in group:
            k+=1
        if item in all_items:
            n+=1

    for item in all_items:
        if item in group: 
            K+=1
    
    expected = (n*K)/N
    fold = k/expected
    p_value = 0
    total_combinations = nCr(N, n)
    for x in range(k, min(n, K) + 1):
        # P(X = x)
        prob_x = (nCr(K, x) * nCr(N - K, n - x)) / total_combinations
        p_value += prob_x
    rez = {
        "N": N,
        "K": K,
        "n": n,
        "k": k,
        "expected": expected,
        "fold": fold,
        "p_value": p_value
    }
    return rez 

    
    



def count_significant(all_items, selected_items, groups, alpha=0.05):
    """Vrne število skupin s p_value < alpha."""
    count = 0
    for group in groups:
        res = enrichment(all_items, selected_items, group)
        if res["p_value"] < alpha:
            count += 1
    return count
    


if __name__ == "__main__":
    all_items = set(range(12))
    selected = {0, 1, 2, 3, 11}
    print(enrichment(all_items, selected, set(range(5))))
