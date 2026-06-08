# Stopnja učenja in konvergenca

## Opis

Implementiraj univariatno linearno regresijo z gradientnim sestopom in razišči,
kako stopnja učenja (`lr`) vpliva na konvergenco.

## Naloga

V datoteki `lin_reg.py` dopolni funkcijo:

```python
def train(xs, ys, lr, n_epochs):
    ...
```

Funkcije `predict`, `mse_loss` in `make_data` so že podane.

## Vrnjena vrednost

- `train(xs, ys, lr, n_epochs)` naj z gradientnim sestopom (začni pri `w = 0`,
  `b = 0`) vrne nauceni `(w, b)`. Gradienta:
  `dL/dw = 2*mean((p - y) * x)`, `dL/db = 2*mean(p - y)`.

## Raziščite sami

Uporabi podatke iz poglavja (`random.seed(42)`, `n = 5`, `y = 2x - 1 + šum`).
Za vsako stopnjo učenja iz množice `{0.001, 0.01, 0.1, 1.0, 10.0}` zaženi
`train(xs, ys, lr, n_epochs=200)` in zabeleži končni `w`, `b` ter morebitno
napako. Določi največjo stopnjo, pri kateri učenje še konvergira blizu `w ≈ 2`
in `b ≈ -1`, ter najmanjšo, pri kateri pride do divergence (npr.
`OverflowError`). Na kratko razloži, zakaj prevelik korak destabilizira učenje.

```python
xs, ys = make_data()
for lr in [0.001, 0.01, 0.1, 1.0, 10.0]:
    try:
        w, b = train(xs, ys, lr, n_epochs=200)
        print(f"lr={lr:<6} w={w:.3f} b={b:.3f} loss={mse_loss(xs, ys, w, b):.3f}")
    except OverflowError:
        print(f"lr={lr:<6} OverflowError (divergenca)")
```

## Testiranje

```bash
python -m unittest -v test_lin_reg
```
