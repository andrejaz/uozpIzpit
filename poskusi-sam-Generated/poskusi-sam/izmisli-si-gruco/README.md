# Izmisli si gručo

## Opis

Brez uporabe metod gručenja sam izberi nekaj primerov, ki se ti zdijo zanimivi,
in jih obravnavaj kot gručo. Nato gručo razloži z rangiranjem (zveznih) značilk
glede na to, kako močno ločijo izbrano skupino od preostalih primerov.

## Naloga

V datoteki `razlaga.py` dopolni funkciji:

```python
def t_statistic(group_vals, other_vals):
    ...


def rank_features_ttest(X, mask):
    ...
```

Funkciji `load_data` (prebere `cluster-data.csv`) in podatkovna množica sta
podani.

## Podatki

Datoteko `cluster-data.csv` po potrebi ustvariš z:

```bash
python generate_data.py
```

## Vrnjena vrednost

- `t_statistic(group_vals, other_vals)` naj vrne (Welchovo) t-statistiko med
  dvema skupinama vrednosti.
- `rank_features_ttest(X, mask)` naj vrne indekse atributov, urejene padajoče po
  absolutni vrednosti t-statistike (skupina = primeri, kjer je `mask` True).

## Raziščite sami

Ročno izberi 10–20 primerov, ki se ti zdijo zanimivi, in jih obravnavaj kot
gručo. Z rangiranjem atributov razloži skupino in preveri, ali rangirane
značilke povedo smiselno zgodbo. (Pri svojem podatkovnem naboru bi tu uporabil
domensko znanje.)

## Testiranje

```bash
python -m unittest -v test_razlaga
```
