# Kaj vidi posamezna metoda? (MDS in t-SNE)

## Opis

Na majhnih množicah sta MDS in t-SNE pogosto podobna. V tej nalogi ustvari več
dobro ločenih skupin v visokorazsežnem prostoru, ju vloži z obema metodama in
primerjaj, kako dobro vsaka ohrani ločenost skupin.

## Naloga

V datoteki `embeddings.py` dopolni funkciji:

```python
def make_groups_highdim(n_per, dim, sep, seed=0):
    ...


def silhouette(X, labels):
    ...
```

Za raziskovanje sta na voljo `mds` (v `mds.py`) in `tsne` (v `tsne_simple.py`).

## Vrnjena vrednost

- `make_groups_highdim(n_per, dim, sep, seed)` naj ustvari tri Gaussove skupine
  (vsaka s `n_per` točkami) v `dim`-razsežnem prostoru, ločene z razmikom `sep`,
  in vrne `(X, labels)`.
- `silhouette(X, labels)` naj vrne povprečno silhueto razbitja (cenilka
  ločljivosti skupin, vrednost blizu 1 pomeni dobro ločene skupine).

## Raziščite sami

Ustvari tri dobro ločene skupine v desetrazsežnem prostoru in jih vloži z MDS in
t-SNE. Nato postopoma zmanjšuj razmik med skupinami in primerjaj, pri kateri
stopnji prekrivanja posamezna metoda še uspe ločiti skupine. Kaj ohranja MDS in
kaj t-SNE?

## Testiranje

```bash
python -m unittest -v test_embeddings
```
