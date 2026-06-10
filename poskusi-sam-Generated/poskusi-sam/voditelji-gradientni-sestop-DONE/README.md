# Metoda voditeljev z gradientnim sestopom

## Opis

Implementiraj poenostavljeno metodo voditeljev, kjer so parametri modela
koordinate voditeljev `v_1, ..., v_K`. Kriterijska funkcija je povprečna
kvadratna razdalja vsakega primera do najbližjega voditelja:

```text
J(v_1, ..., v_K) = (1/n) Σ_i min_k ||x_i - v_k||^2
```

## Naloga

V datoteki `voditelji.py` dopolni funkciji:

```python
def assign(X, leaders):
    ...


def train_leaders(X, k, lr=0.5, steps=300, seed=0):
    ...
```

## Vrnjena vrednost

- `assign(X, leaders)` naj vrne indeks najbližjega voditelja za vsak primer.
- `train_leaders(X, k, lr, steps, seed)` naj voditelje inicializira na naključne
  primere in jih z gradientnim sestopom premika proti zmanjšanju `J`. Vrne
  matriko voditeljev oblike `(k, d)`.

## Raziščite sami

Ustvari dvodimenzionalne podatke z znanimi gručami, voditelje inicializiraj
naključno in jih uči z gradientnim sestopom. Po nekaj korakih izriši podatke in
trenutne položaje voditeljev. Primerjaj rezultat s klasično metodo voditeljev
(priredi najbližjemu, nato povprečje).

## Testiranje

```bash
python -m unittest -v test_voditelji
```
