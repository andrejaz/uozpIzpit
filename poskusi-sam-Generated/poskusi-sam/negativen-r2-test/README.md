# Negativen R² na testni množici

## Opis

Mera `R²` pove, kolikšen delež variance pojasni model glede na osnovni
(konstantni) model, ki napoveduje povprečje. Na testni množici je lahko `R²`
tudi negativen – takrat je model slabši od konstantne napovedi.

## Naloga

V datoteki `r2.py` dopolni funkcije:

```python
def r2_score(y_true, y_pred):
    ...


def polynomial_features(x, degree):
    ...


def train_test_split(X, y, test_ratio, seed=0):
    ...
```

Funkcija `fit_least_squares(X, y)` (navadna linearna regresija) je že podana.

## Vrnjena vrednost

- `r2_score(y_true, y_pred)` naj vrne `1 - SS_res / SS_tot`, kjer je
  `SS_tot` vsota kvadratov odstopanj od povprečja `y_true`.
- `polynomial_features(x, degree)` naj za vektor `x` vrne matriko stolpcev
  `[x, x^2, ..., x^degree]` oblike `(n, degree)`.
- `train_test_split(X, y, test_ratio, seed)` naj naključno razdeli podatke in
  vrne `(X_train, X_test, y_train, y_test)`.

## Raziščite sami

Pripravi majhno univariatno učno množico, jo razdeli na učno in testno (npr.
70 % / 30 %) ter z linearno regresijo zgradi modele s polinomsko razširitvijo
različnih stopenj. Poišči primer delitve ali stopnje polinoma, pri kateri je
testni `R²` negativen. Zakaj model napoveduje slabše od konstantnega modela?

## Testiranje

```bash
python -m unittest -v test_r2
```
