# Kritična stopnja učenja

## Opis

Poišči največjo stopnjo učenja, pri kateri gradientni sestop pri linearni
regresiji še ostane stabilen.

## Naloga

V datoteki `lin_reg.py` dopolni funkciji:

```python
def diverged(xs, ys, lr, n_epochs):
    ...


def find_critical_lr(xs, ys, n_epochs=200, start=1e-3, factor=2.0):
    ...
```

Funkcija `train` je že podana (vrne `(w, b)`).

## Vrnjena vrednost

- `diverged(xs, ys, lr, n_epochs)` naj vrne `True`, če učenje podivja (uteži
  postanejo `nan`/`inf` ali izguba zelo naraste).
- `find_critical_lr(...)` naj s podvajanjem stopnje učenja poišče prvo nestabilno
  vrednost, nato pa z bisekcijo med zadnjo stabilno in prvo nestabilno vrednostjo
  oceni največjo stabilno stopnjo učenja in jo vrne.

## Raziščite sami

Uporabi podatke iz poglavja. Začni pri konvergentni stopnji učenja (npr.
`0.001`) in jo iterativno podvajaj, dokler učenje z `n_epochs = 200` ne
divergira. Nato z bisekcijo zoži interval med zadnjo stabilno in prvo
nestabilno vrednostjo ter oceni največjo stabilno stopnjo učenja.

## Testiranje

```bash
python -m unittest -v test_lin_reg
```
