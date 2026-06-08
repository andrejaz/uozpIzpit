# Regularizacijska pot in nestabilnost izbora značilk

## Opis

Regularizacijska pot je zaporedje ocenjenih uteži pri naraščajočih vrednostih
stopnje regularizacije `λ`. Pri L1 (LASSO) nekatere uteži postopoma postanejo
natanko nič.

## Naloga

V datoteki `regularizacija.py` dopolni funkcijo:

```python
def regularization_path(X, y, reg, lambdas):
    ...
```

Funkciji `fit` (regularizirana linearna regresija) in `make_data` sta že podani.
`make_data` ustvari množico s tremi informativnimi, tremi nepomembnimi in tremi
močno koreliranimi (kopijami prvega informativnega) atributi.

## Vrnjena vrednost

- `regularization_path(X, y, reg, lambdas)` naj za vsako vrednost `λ` iz
  `lambdas` nauči model z regularizacijo `reg` (`"l1"` ali `"l2"`) in vrne
  matriko uteži oblike `(len(lambdas), n_features)`.

## Raziščite sami

Za vsako vrednost `λ` nauči model z L1 ali L2 regularizacijo in izriši
odvisnost vrednosti uteži posameznih atributov od stopnje regularizacije. Pri L1
spremljaj, katere uteži postanejo ničelne in kdaj. Poskus ponovi z vsaj tremi
različnimi podatkovnimi množicami (spreminjaj `seed` v `make_data`).

## Testiranje

```bash
python -m unittest -v test_regularizacija
```
