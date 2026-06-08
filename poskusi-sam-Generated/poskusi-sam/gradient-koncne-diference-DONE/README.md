# Gradient s končnimi diferencami

## Opis

Za funkcijo štirih parametrov

```text
L(a, b, c, d) = (a*b + c) * d
```

oceni vse štiri parcialne odvode z metodo končnih diferenc in jih primerjaj z
analitičnim gradientom.

## Naloga

V datoteki `gradient.py` dopolni funkcije:

```python
def numerical_gradient(func, params, h=1e-5):
    ...


def analytic_gradient(a, b, c, d):
    ...
```

## Vrnjena vrednost

- `numerical_gradient(func, params, h)` naj za zaporedje `params = (a, b, c, d)`
  vrne seznam parcialnih odvodov funkcije `func` glede na vsak parameter
  (vsakega posebej zmotimo za `h`).
- `analytic_gradient(a, b, c, d)` naj vrne `(dL/da, dL/db, dL/dc, dL/dd)`, torej
  `(b*d, a*d, d, a*b + c)`.

## Raziščite sami

Za vrednosti `a = 2, b = 3, c = 10, d = 2` izračunaj numerični gradient in ga
primerjaj z analitičnim. Pri teh vrednostih je pravi gradient `(6, 4, 2, 16)`,
saj velja `dL/dd = a*b + c = 16`.

> Opomba: v učbeniku je pri tej nalogi naveden gradient `(6, 4, 2, 4)`, kar
> ustreza glavnemu zgledu poglavja z drugačnimi vrednostmi (`b = -3, d = -2`,
> kjer je `a*b + c = 4`). Tu uporabljamo vrednosti naloge in pravilen rezultat
> `(6, 4, 2, 16)`.

```python
print(analytic_gradient(2, 3, 10, 2))
print(numerical_gradient(L, (2, 3, 10, 2)))
```

## Testiranje

```bash
python -m unittest -v test_gradient
```
