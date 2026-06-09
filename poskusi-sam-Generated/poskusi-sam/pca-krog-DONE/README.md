# Zakaj PCA ne vidi kroga?

## Opis

PCA poišče smer največje variance, kar ni vedno smiselno za podatke z očitno
geometrijsko strukturo. Pri točkah na krožnici nobena smer ni izstopajoča.

## Naloga

V datoteki `pca.py` dopolni funkcijo:

```python
def explained_variance_ratio(X, v):
    ...
```

Funkciji `first_principal_component(X)` in `make_circle(n)` sta že podani.

## Vrnjena vrednost

- `explained_variance_ratio(X, v)` naj vrne delež razložene variance v smeri `v`:
  varianco projekcije centriranih podatkov na enotni vektor `u = v / ||v||`,
  deljeno s celotno varianco (vsota varianc po stolpcih).

## Raziščite sami

Generiraj točke na krožnici (ali polkrožnici). Izračunaj PCA in podatke
projiciraj na prvo komponento. Rezultat vizualiziraj ter razloži, zakaj PCA
kljub očitni geometrijski strukturi ne najde uporabne enodimenzionalne
predstavitve. (Pri polni krožnici prva komponenta razloži približno 50 %
variance – vse smeri so enakovredne.)

## Testiranje

```bash
python -m unittest -v test_pca
```
