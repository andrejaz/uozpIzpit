# Odvod in minimum kvadratne funkcije

## Opis

V tej nalogi boste uporabili gradientni sestop za preprosto kvadratno funkcijo:

```python
f(a) = a**2 - 10*a + 28
```

Najprej izračunajte odvod funkcije pri dani vrednosti `a`, nato pa z več zaporednimi koraki gradientnega sestopa približno poiščite vrednost `a`, pri kateri ima funkcija minimum.

## Naloga

V datoteki `quadratic_gradient_descent.py` implementirajte funkciji:

```python
def derivative(a):
    ...


def find_minimum(start, learning_rate, steps):
    ...
```

## Argumenti

- `a`: vrednost parametra, pri kateri izračunate odvod funkcije.
- `start`: začetna vrednost parametra pri gradientnem sestopu.
- `learning_rate`: hitrost učenja oziroma velikost koraka.
- `steps`: število korakov gradientnega sestopa.

## Vrnjena vrednost

Funkcija `derivative(a)` naj vrne vrednost odvoda funkcije `f(a)`.

Funkcija `find_minimum(start, learning_rate, steps)` naj vrne končno vrednost parametra po `steps` korakih gradientnega sestopa.

## Primer

```python
result = find_minimum(6, 0.1, 3)
print(round(result, 3))
```

Pričakovani rezultat:

```text
5.512
```

## Testiranje

Teste poženete z ukazom:

```bash
python -m unittest -v test_quadratic_gradient_descent
```
