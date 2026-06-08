# Odpornost vložitev na šum

## Opis

Razišči, kako napaka v eni od razdalj vpliva na rekonstruirani zemljevid (MDS).

## Naloga

V datoteki `mds.py` dopolni funkciji:

```python
def perturb_distance(distances, pair, delta):
    ...


def total_movement(pos_a, pos_b):
    ...
```

Funkciji `mds` in `stress` ter slovar `DISTANCES` so že podani.

## Vrnjena vrednost

- `perturb_distance(distances, pair, delta)` naj vrne **kopijo** slovarja razdalj,
  v kateri je razdalja za `pair` (in po potrebi za simetrični par) povečana za
  `delta`. Izvirni slovar ostane nespremenjen.
- `total_movement(pos_a, pos_b)` naj vrne vsoto evklidskih razdalj med položaji
  istih mest v dveh vložitvah `pos_a` in `pos_b`.

## Raziščite sami

Izberi eno od razdalj in ji umetno dodaj veliko napako (npr. razdaljo
Ljubljana–Koper povečaj za 100 km). Ponovno izračunaj MDS-vložitev in primerjaj
položaje mest pred in po spremembi. Poskus ponovi za več velikosti napake in za
različne pare mest.

> Opomba: vložitve MDS so določene le do rotacije, zrcaljenja in premika, zato je
> za pošteno primerjavo položajev pogosto potrebna poravnava (npr. Procrustes).
> `total_movement` poravnave ne izvaja.

## Testiranje

```bash
python -m unittest -v test_mds
```
