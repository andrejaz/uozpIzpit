# Numerični odvod na zahtevnejši funkciji

## Opis

V tej nalogi boste uporabili numerični približek odvoda. Funkcija, ki jo minimiziramo, je že podana in uporablja trigonometrične ter hiperbolične funkcije:

```python
def objective(x):
    return (
        math.sin(1.7 * x)
        + 0.4 * math.cos(3.1 * x)
        + 0.15 * (x - 1.5) ** 2
        + math.tanh(x - 0.5)
    )
```

Odvoda ne računajte ročno. Ocenite ga z metodo končnih diferenc in ga uporabite v gradientnem sestopu. Funkcija ni nujno konveksna, zato naloga preverja obnašanje za podane začetne vrednosti na omejenem intervalu.

## Naloga

V datoteki `numerical_gradient_descent.py` implementirajte funkciji:

```python
def numerical_derivative(x, h=0.0001):
    ...


def find_minimum(start, learning_rate, steps, h=0.0001):
    ...
```

## Argumenti

- `x`: vrednost, pri kateri ocenite odvod funkcije.
- `h`: majhen premik, ki ga uporabite pri metodi končnih diferenc.
- `start`: začetna vrednost parametra pri gradientnem sestopu.
- `learning_rate`: hitrost učenja oziroma velikost koraka.
- `steps`: število korakov gradientnega sestopa.

## Vrnjena vrednost

Funkcija `numerical_derivative(x, h)` naj vrne numerični približek odvoda funkcije `objective` pri vrednosti `x`.

Funkcija `find_minimum(start, learning_rate, steps, h)` naj vrne končno vrednost parametra po `steps` korakih gradientnega sestopa.

## Primer

```python
result = numerical_derivative(0.0)
print(round(result, 3))
```

Pričakovani rezultat:

```text
2.036
```

## Testiranje

Teste poženete z ukazom (iz mape naloge):

```bash
python -m unittest test_numerical_gradient_descent.py
```