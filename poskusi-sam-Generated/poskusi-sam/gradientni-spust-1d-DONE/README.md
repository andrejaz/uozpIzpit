# Gradientni spust v eni spremenljivki

## Opis

Implementiraj gradientni spust za iskanje lokalnega minimuma funkcije

```text
f(x) = x^4 - 3x^3 + 2
```

na intervalu `[0, 3]`. Odvod izračunaj numerično z metodo končnih diferenc.

## Naloga

V datoteki `gradientni_spust.py` dopolni funkcije:

```python
def numerical_derivative(func, x, h=1e-5):
    ...


def gradient_descent(func, x0, lr=0.01, steps=1000, h=1e-5):
    ...
```

## Vrnjena vrednost

- `numerical_derivative(func, x, h)` naj vrne numerični odvod (končne diference).
- `gradient_descent(func, x0, lr, steps, h)` naj iz začetne vrednosti `x0`
  iterativno popravlja `x` v smeri negativnega odvoda
  (`x = x - lr * f'(x)`) in vrne končni `x`.

## Raziščite sami

Funkcija ima na intervalu `[0, 3]` lokalni minimum pri `x = 9/4 = 2.25`
(tam velja `f'(x) = 4x^3 - 9x^2 = 0`). Preizkusi različne začetne vrednosti
`x0` v `[0, 3]` in opazuj, kam konvergira postopek.

```python
for x0 in [0.5, 1.0, 2.0, 2.9]:
    x = gradient_descent(f, x0, lr=0.01, steps=20000)
    print(f"x0={x0}  ->  x={x:.4f}  f(x)={f(x):.4f}")
```

## Testiranje

```bash
python -m unittest -v test_gradientni_spust
```
