# Gradientni sestop v dveh spremenljivkah

## Opis

Poišči minimum funkcije

```text
f(a, b) = a^2 + b^2 + a*b - 6a - 4b
```

z gradientnim sestopom. Parametra `a` in `b` predstavi z vozlišči `Value`,
gradiente izračunaj s strojnim odvajanjem (`backward()`).

## Naloga

V datoteki `gradientni_sestop_2d.py` dopolni funkcije:

```python
def f_value(a, b):
    ...


def train_2d(lr=0.1, steps=50, a0=0.0, b0=0.0):
    ...


def analytic_min():
    ...
```

## Vrnjena vrednost

- `f_value(a, b)` naj za vozlišči `Value` vrne vozlišče za `f(a, b)`.
- `train_2d(...)` naj začne pri `(a0, b0)`, naredi `steps` korakov gradientnega
  sestopa s hitrostjo učenja `lr` in vrne `(a, b, f)` kot števila (`float`).
- `analytic_min()` naj vrne analitično rešitev `(a*, b*, f*)`.

## Raziščite sami

Analitično reši `∂f/∂a = 0` in `∂f/∂b = 0`:

```text
2a + b - 6 = 0
a + 2b - 4 = 0   ->   a* = 8/3,  b* = 2/3,  f* = -28/3
```

Začni pri `a = 0, b = 0`, naredi 50 (ali več) korakov s hitrostjo učenja `0.1` in
preveri, da se rezultat ujema z analitično rešitvijo. Na koncu z metodo končnih
diferenc numerično oceni parcialna odvoda v končni točki in ju primerjaj z
`a.grad` in `b.grad`.

## Testiranje

```bash
python -m unittest -v test_gradientni_sestop_2d
```
