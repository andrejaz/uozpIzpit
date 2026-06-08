# Lov na največjo silhueto

## Opis

Silhueta je cenilka kakovosti razbitja, ki upošteva kohezijo znotraj gruč in
ločljivost med njimi. V tej nalogi razišči, ali metoda voditeljev tudi v
podatkih brez prave strukture vedno predlaga neko število gruč.

## Naloga

V datoteki `silhueta.py` dopolni funkciji:

```python
def silhouette_score(X, labels):
    ...


def best_k(X, ks, seed=0):
    ...
```

Funkcija `kmeans` je podana.

## Vrnjena vrednost

- `silhouette_score(X, labels)` naj vrne povprečno silhueto razbitja
  (`s_i = (b_i - a_i) / max(a_i, b_i)`).
- `best_k(X, ks, seed)` naj za vsako `k` iz `ks` izvede metodo voditeljev,
  izračuna silhueto in vrne `k` z največjo vrednostjo.

## Raziščite sami

Ustvari dvodimenzionalne podatke brez prave strukture (npr. 200 točk iz ene
Gaussove porazdelitve). Za `K = 2, 3, ..., 8` izvedi metodo voditeljev, za vsako
vrednost izračunaj silhueto in izberi `K` z največjo vrednostjo. Ali metoda
kljub odsotnosti prave strukture vedno predlaga neko število gruč? Kaj to pove o
silhueti kot dokazu obstoja gruč?

## Testiranje

```bash
python -m unittest -v test_silhueta
```
