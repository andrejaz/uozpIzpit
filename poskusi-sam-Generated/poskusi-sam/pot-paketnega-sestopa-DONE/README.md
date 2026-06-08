# Pot paketnega sestopa

## Opis

Spremljaj pot paketnega (mini-batch) gradientnega sestopa v prostoru parametrov
`(w, b)`.

## Naloga

V datoteki `lin_reg.py` dopolni funkcijo:

```python
def train_batches_with_history(xs, ys, lr, n_epochs, batch_size, seed=0):
    ...
```

## Vrnjena vrednost

- `train_batches_with_history(...)` naj vrne seznam parov `(w, b)` dolžine
  `n_epochs + 1` (prvi je `(0.0, 0.0)`), kjer en korak ustreza enemu paketu.

## Raziščite sami

Uporabi podatke za univariatno regresijo. Za isti začetni model zaženi učenje s
paketi (`batch_size = 10`) in `n_epochs = 200` pri stopnjah učenja `0.001` in
`0.1`, nato izriši obe poti v ravnini `(w, b)`. Kaj opaziš? Pot paketnega
sestopa je zaradi šumnih ocen gradienta bolj cik-cak kot pri polnem paketu.

## Testiranje

```bash
python -m unittest -v test_lin_reg
```
