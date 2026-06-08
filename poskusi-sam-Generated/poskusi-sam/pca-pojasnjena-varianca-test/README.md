# Pojasnjena varianca na testni množici

## Opis

PCA naučiš na učni množici, nato pa testne podatke projiciraš na naučene
komponente in na njih izračunaš delež pojasnjene variance. Standardizacijo
izvedeš na učni množici in iste parametre uporabiš na testni.

## Naloga

V datoteki `pca.py` dopolni funkciji:

```python
def pca_fit(X_train_std):
    ...


def explained_on_test(X_test_std, components, k):
    ...
```

Funkcije `make_data`, `standardize_fit` in `standardize_apply` so že podane.

## Vrnjena vrednost

- `pca_fit(X_train_std)` naj vrne matriko komponent (vrstice so glavne
  komponente, urejene po padajoči lastni vrednosti).
- `explained_on_test(X_test_std, components, k)` naj projicira testne podatke na
  prvih `k` komponent in vrne delež pojasnjene variance: vsoto varianc projekcij
  deljeno s celotno varianco testnih podatkov.

## Raziščite sami

Standardiziraj podatke (parametre nauči na učni množici!) in jih večkrat
naključno razdeli na učno in testno. PCA nauči na učni množici, nato na testni
izračunaj delež variance, ki ga pojasnijo prva, prvi dve in prve tri komponente.
Poskus ponovi za velikosti učne množice 10, 20, 50, 100, 200 in rezultate
povpreči.

## Testiranje

```bash
python -m unittest -v test_pca
```
