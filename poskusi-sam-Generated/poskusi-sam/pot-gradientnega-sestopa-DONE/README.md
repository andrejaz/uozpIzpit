# Pot gradientnega sestopa

## Opis

Spremljaj pot, ki jo gradientni sestop opravi v prostoru parametrov `(w, b)`.

## Naloga

V datoteki `lin_reg.py` dopolni funkcijo:

```python
def train_with_history(xs, ys, lr, n_epochs):
    ...
```

## Vrnjena vrednost

- `train_with_history(xs, ys, lr, n_epochs)` naj vrne seznam parov `(w, b)`
  dolžine `n_epochs + 1`. Prvi element naj bo začetno stanje `(0.0, 0.0)`, vsak
  naslednji pa stanje po enem koraku gradientnega sestopa.

## Raziščite sami

Uporabi podatke iz poglavja. Za isti začetni model zaženi učenje z
`n_epochs = 100` pri stopnjah učenja `0.001` in `0.1`, nato v ravnini `(w, b)`
izriši obe poti (točke poveži v zaporedju). Opiši, zakaj majhna stopnja učenja
vodi do gladke poti proti minimumu, prevelika pa do cik-cak gibanja čez dolino
izgube.

```python
import matplotlib.pyplot as plt
xs, ys = make_data()
for lr in [0.001, 0.1]:
    path = train_with_history(xs, ys, lr, n_epochs=100)
    ws, bs = zip(*path)
    plt.plot(ws, bs, marker="o", label=f"lr={lr}")
plt.xlabel("w"); plt.ylabel("b"); plt.legend(); plt.savefig("pot.pdf")
```

## Testiranje

```bash
python -m unittest -v test_lin_reg
```
