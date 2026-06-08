# Gruče držav na mreži podobnosti

## Opis

Iz podatkov zgradi omrežje, kjer sta dve državi povezani, če je njuna kosinusna
razdalja (na standardiziranih značilkah) manjša od izbranega praga. Na omrežju
nato poišči skupnosti z razširjanjem oznak (label propagation).

## Naloga

V datoteki `mreza.py` dopolni funkcije:

```python
def cosine_distance(a, b):
    ...


def build_graph(X, threshold):
    ...


def label_propagation(adj, seed=0, max_iter=100):
    ...
```

Funkcija `load_data` (prebere `hdi.csv`) je podana.

## Podatki

Datoteko `hdi.csv` po potrebi ustvariš z:

```bash
python generate_data.py
```

## Vrnjena vrednost

- `cosine_distance(a, b)` naj vrne `1 - kosinusna podobnost`.
- `build_graph(X, threshold)` naj vrne slovar sosednosti `{i: množica sosedov}`,
  kjer sta `i` in `j` povezana, če je `cosine_distance(X[i], X[j]) <= threshold`.
- `label_propagation(adj, ...)` naj z razširjanjem oznak vrne slovar
  `{vozlišče: oznaka skupnosti}`.

## Raziščite sami

Pri tolikšnem številu značilk je kosinusna razdalja pogosto smiselnejša od
evklidske (manj občutljiva na skupno velikost vektorjev). S smiselno izbranim
pragom zgradi omrežje, da dobiš tri do pet skupin, in rezultate prikaži na
barvnem zemljevidu sveta.

## Testiranje

```bash
python -m unittest -v test_mreza
```
