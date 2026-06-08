# Preveč skupin, premalo statistike

## Opis

Analiza obogatenosti meri, ali se elementi neke skupine v izboru pojavljajo
pogosteje, kot bi pričakovali po naključju (hipergeometrijska porazdelitev). Pri
mnogih hkratnih testih (večkratno testiranje) pa nekatere skupine izpadejo
značilne zgolj po naključju.

## Naloga

V datoteki `obogatenost.py` dopolni funkciji:

```python
def enrichment(all_items, selected_items, group):
    ...


def count_significant(all_items, selected_items, groups, alpha=0.05):
    ...
```

## Vrnjena vrednost

- `enrichment(all_items, selected_items, group)` naj vrne slovar z vrednostmi
  `N, K, n, k, expected, fold, p_value`, kjer je `p_value` verjetnost
  `P(X >= k)` po hipergeometrijski porazdelitvi (vzorčenje brez vračanja):
  `N = |all_items|`, `K = |group ∩ all_items|`, `n = |selected ∩ all_items|`,
  `k = |group ∩ selected|`, `expected = n*K/N`, `fold = k/expected`.
- `count_significant(...)` naj vrne število skupin s `p_value < alpha`.

## Raziščite sami

Ustvari 100 primerov in jih naključno razdeli v dve skupini po 50; eno skupino
obravnavaj kot izbor, drugo kot preostanek. Nato generiraj `m` naključnih skupin
in za vsako izračunaj obogatenost. Preštej, koliko skupin ima `p < 0.05`. Poskus
ponovi za `m = 100` in `m = 1000`. Kaj se zgodi s številom obogatenih skupin, ko
povečuješ `m`? (Razmisli o popravkih za večkratno testiranje, npr.
Bonferronijevem.)

## Testiranje

```bash
python -m unittest -v test_obogatenost
```
