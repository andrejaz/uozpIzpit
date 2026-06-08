# Ali PCA odkrije skupine povezanih atributov?

## Opis

Če izvirajo skupine atributov iz skupnih skritih spremenljivk, jih PCA prepozna
kot smeri velike variance. Uteži atributov iste skupine na pripadajoči glavni
komponenti so podobne.

## Naloga

V datoteki `pca.py` dopolni funkciji:

```python
def top_components(X, k):
    ...


def explained_variance_ratios(X):
    ...
```

Funkcija `make_grouped_data(n, noise, seed)` (9 atributov v 3 skupinah iz 3
skritih spremenljivk) je že podana.

## Vrnjena vrednost

- `top_components(X, k)` naj vrne matriko oblike `(k, d)` z utežmi prvih `k`
  glavnih komponent (lastni vektorji kovariančne matrike, urejeni po padajoči
  lastni vrednosti).
- `explained_variance_ratios(X)` naj vrne deleže razložene variance vseh
  komponent, urejene padajoče.

## Raziščite sami

Za prve tri glavne komponente izpiši uteži posameznih atributov in jih prikaži z
grafom. Preveri, ali imajo atributi iz iste skupine podobne uteži na isti
komponenti. Nato povečaj šum in opazuj, kako se spreminjata delež pojasnjene
variance in interpretabilnost komponent.

## Testiranje

```bash
python -m unittest -v test_pca
```
