# Velikost paketa in konvergenca

## Opis

Implementiraj paketno (mini-batch) učenje linearne regresije in razišči vpliv
velikosti paketa na konvergenco.

## Naloga

V datoteki `lin_reg.py` dopolni funkcijo:

```python
def train_batches(xs, ys, lr, n_epochs, batch_size, seed=0):
    ...
```

## Vrnjena vrednost

- `train_batches(...)` naj v vsaki iteraciji naključno izbere `batch_size`
  primerov, na njih izračuna gradient in posodobi `(w, b)`. Vrne `(w, b)`.
  Za naključno izbiro uporabi `numpy.random.default_rng(seed)`.

## Raziščite sami

Uporabi podatke iz poglavja (`random.seed(42)`, `n = 1000`,
`y = 2x + 1 + šum`). Za vsako velikost paketa iz `{1, 5, 10, 50, 100, 1000}`
zaženi učenje z `learning_rate = 0.01` in `n_epochs = 500`. Zabeleži končne
parametre in izgubo na celotni učni množici ter primerjaj poti izgube. Katera
velikost paketa povzroči najbolj nestanovitno učenje? Katera konvergira najhitreje?
Zakaj majhni paketi vodijo do bolj šumnih ocen gradienta?

## Testiranje

```bash
python -m unittest -v test_lin_reg
```
