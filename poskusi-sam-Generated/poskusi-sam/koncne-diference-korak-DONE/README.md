# Korak pri končnih diferencah

## Opis

Odvod funkcije lahko namesto analitično ocenimo numerično z metodo končnih
diferenc:

```text
f'(a) ≈ (f(a + h) - f(a)) / h
```

V tej nalogi raziščeš, kako izbira koraka `h` vpliva na natančnost ocene. Pri
prevelikem `h` je ocena nenatančna zaradi odreza (linearna aproksimacija), pri
premajhnem `h` pa se pojavijo numerične težave zaradi omejene natančnosti
plavajoče vejice (odštevanje skoraj enakih števil).

## Naloga

V datoteki `koncne_diference.py` dopolni funkcije:

```python
def forward_difference(func, a, h):
    ...


def analytic_derivative(a):
    ...
```

Funkcija `f(a) = a^2 - 10a + 28` je že podana.

## Vrnjena vrednost

- `forward_difference(func, a, h)` naj vrne `(func(a + h) - func(a)) / h`.
- `analytic_derivative(a)` naj vrne analitični odvod funkcije `f`, torej `2a - 10`.

## Raziščite sami

Za funkcijo `f(a) = a^2 - 10a + 28` izračunaj odvod v točki `a = 6` z metodo
končnih diferenc. Preizkusi vrednosti `h = 1e-1, 1e-2, 1e-4, 1e-8` in rezultate
primerjaj z analitičnim odvodom (`f'(6) = 2`). Opiši, pri katerih vrednostih `h`
je ocena najboljša in kdaj se začnejo pojavljati numerične težave.

```python
for h in [1e-1, 1e-2, 1e-4, 1e-8, 1e-12]:
    approx = forward_difference(f, 6.0, h)
    print(f"h={h:.0e}  ocena={approx:.8f}  napaka={abs(approx - 2.0):.2e}")
```

## Testiranje

```bash
python -m unittest -v test_koncne_diference
```
