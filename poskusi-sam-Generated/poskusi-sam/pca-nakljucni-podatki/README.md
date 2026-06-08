# Koliko variance pojasni prva komponenta pri naključnih podatkih?

## Opis

Pri popolnoma naključnih (nekoreliranih) standardiziranih podatkih z `d` atributi
naj bi prva glavna komponenta v povprečju pojasnila približno `1/d` variance.
Zaradi končnih vzorcev pa je pojasnjena varianca nekoliko večja.

## Naloga

V datoteki `pca.py` dopolni funkciji:

```python
def explained_variance_first(X):
    ...


def mean_explained_first(n, d, reps, seed=0):
    ...
```

## Vrnjena vrednost

- `explained_variance_first(X)` naj vrne delež variance, ki ga pojasni prva
  glavna komponenta (`največja lastna vrednost / vsota lastnih vrednosti`
  kovariančne matrike).
- `mean_explained_first(n, d, reps, seed)` naj `reps`-krat ustvari `n x d`
  standardizirano množico iz standardne normalne porazdelitve in vrne povprečje
  deležev variance prve komponente.

## Raziščite sami

Ustvari množico z desetimi atributi iz standardne normalne porazdelitve,
standardiziraj in izračunaj PCA. Izmeri delež variance prve komponente in ga
primerjaj s teoretično vrednostjo `1/10`. Poskus ponovi za različne velikosti
vzorca (npr. 20, 50, 100, 500, 5000) in nariši graf povprečne pojasnjene
variance v odvisnosti od velikosti vzorca. Ali se z večanjem množice približuje
pričakovani vrednosti?

## Testiranje

```bash
python -m unittest -v test_pca
```
