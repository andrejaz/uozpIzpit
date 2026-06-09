# Varianca projekcije podatkov

## Opis

Dane so centrirane točke v ravnini in smer v prostoru. Varianca projekcij podatkov na to smer pove, kako močno se podatki raztezajo v tej smeri. To je korak, s katerim primerjamo kandidatne smeri, preden poiščemo glavno komponento.

## Naloga

V datoteki `projected_variance.py` implementirajte funkcijo:

```python
def projected_variance(X, direction):
    ...
```

## Argumenti

- `X`: seznam točk, na primer `[(2.0, 0.0), (-1.0, 1.0)]`. Podatki so centrirani.
- `direction`: par `(dx, dy)`, ki poda smer projekcije. Ni nujno, da je enotski vektor.

## Vrnjena vrednost

Funkcija naj vrne varianco projekcij vseh točk na podano smer. Uporabite populacijsko varianco (deljenje s številom točk, ne s `n - 1`).

## Primer

```python
X = [(2.0, 0.0), (-2.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
print(round(projected_variance(X, (-1.0, 1.0)), 4))
print(round(projected_variance(X, (-1.0, 3.0)), 4))
```

Pričakovani rezultat:

```text
1.25
0.65
```

## Testiranje

Teste poženete z ukazom (iz mape naloge):

```bash
python -m unittest test_projected_variance.py
```
