# Koliko razdalj potrebujemo za zemljevid?

## Opis

Večrazsežno lestvičenje (MDS) primere razporedi v ravnino tako, da evklidske
razdalje med njimi čim bolje ustrezajo podanim razdaljam. V tej nalogi razišči,
koliko razdalj potrebujemo za smiselno rekonstrukcijo zemljevida.

## Naloga

V datoteki `mds.py` dopolni funkciji:

```python
def stress(positions, distances):
    ...


def mds(distances, dim=2, lr=0.01, epochs=2000, seed=0):
    ...
```

Slovar `DISTANCES` s cestnimi razdaljami med slovenskimi mesti je že podan.

## Vrnjena vrednost

- `stress(positions, distances)` naj vrne vsoto kvadratov razlik med evklidsko
  razdaljo `||pos_a - pos_b||` in podano razdaljo, čez vse pare v `distances`.
  `positions` je slovar `mesto -> položaj` (vektor).
- `mds(distances, dim, lr, epochs, seed)` naj z gradientnim sestopom poišče
  položaje, ki minimizirajo `stress`, in vrne slovar `mesto -> položaj`.

## Raziščite sami

Najprej uporabi vse podane razdalje in z MDS rekonstruiraj zemljevid. Nato
naključno odstranjuj vedno več razdalj (npr. 10 %, 25 %, 50 %, 75 %) in opazuj,
kako se spreminja kakovost vložitve. Kdaj postane problem premalo določen? Ali
obstajajo razdalje, katerih odstranitev škodi bolj kot odstranitev drugih?

## Testiranje

```bash
python -m unittest -v test_mds
```
