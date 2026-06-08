# Analitični PCA in gradientni sestop

## Opis

Glavne komponente so lastni vektorji kovariančne matrike. V tej nalogi
implementiraj analitični PCA in rezultat primerjaj z iskanjem prve komponente z
gradientnim vzponom.

## Naloga

V datoteki `pca.py` dopolni funkcijo:

```python
def analytic_pca(X):
    ...
```

Funkcija `pca_gradient(X)` (iskanje prve komponente z gradientnim vzponom) je že
podana za primerjavo.

## Vrnjena vrednost

- `analytic_pca(X)` naj vrne `(components, eigenvalues, ratios)`, kjer so:
  - `components` matrika, katere vrstice so enotni lastni vektorji kovariančne
    matrike, urejeni po padajoči lastni vrednosti,
  - `eigenvalues` pripadajoče lastne vrednosti (padajoče),
  - `ratios` deleži razložene variance (`eigenvalues / vsota`).

## Raziščite sami

Po korakih iz izpeljave implementiraj analitični PCA: podatke standardiziraj,
izračunaj kovariančno matriko ter poišči lastne vektorje in lastne vrednosti.
Rezultate (smeri glavnih komponent, uteži atributov in delež pojasnjene
variance) primerjaj z implementacijo prek gradientnega vzpona (`pca_gradient`).
Ali se pristopa ujemata? Kje se lahko razlikujeta? Primerjaj tudi hitrost
postopkov.

## Testiranje

```bash
python -m unittest -v test_pca
```
