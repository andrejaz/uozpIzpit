# Uhajanje informacij pri standardizaciji in izboru λ

## Opis

Standardizacijo (in izbor stopnje regularizacije) je treba izvesti **samo na
učnih podatkih**. Če parametre standardizacije izračunaš na celotni množici pred
delitvijo, v oceno uhaja informacija iz testne množice in ocena točnosti je
preveč optimistična.

## Naloga

V datoteki `uhajanje.py` dopolni funkcije:

```python
def standardize_fit(X_train):
    ...


def standardize_apply(X, mean, std):
    ...


def r2_score(y_true, y_pred):
    ...
```

Funkciji `load_data` (prebere `body-fat-brozek.csv`) in `fit_least_squares` sta
podani.

## Podatki

Datoteka `body-fat-brozek.csv` je sintetična različica podatkov o deležu telesne
maščobe (cilj v prvem stolpcu, sledi 14 atributov). Po potrebi jo ustvariš
ponovno z:

```bash
python generate_data.py
```

## Vrnjena vrednost

- `standardize_fit(X_train)` naj vrne `(mean, std)` po stolpcih (izračunana
  **samo** iz učnih podatkov).
- `standardize_apply(X, mean, std)` naj vrne `(X - mean) / std`.
- `r2_score(y_true, y_pred)` naj vrne `1 - SS_res / SS_tot`.

## Raziščite sami

Omeji se na majhno množico (npr. prvih 50 primerov). Iz nje izloči majhno učno
množico (npr. 10 primerov); testna množica so vsi preostali. Primerjaj dva
postopka:

1. **napačni**: standardiziraš vseh 50 primerov pred delitvijo;
2. **pravilni**: najprej ločiš učno in testno množico, parametre standardizacije
   izračunaš le iz učnih podatkov in jih uporabiš na testnih.

Primerjaj testni `R²` in razloži, zakaj mora biti tudi izbira regularizacije
izvedena na učni množici.

## Testiranje

```bash
python -m unittest -v test_uhajanje
```
