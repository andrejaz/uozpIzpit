# Vpliv šuma na oceno uteži

## Opis

Razišči, kako šum v ciljni spremenljivki vpliva na natančnost ocenjenih uteži
pri multivariatni linearni regresiji.

## Naloga

V datoteki `lin_reg.py` dopolni funkcijo:

```python
def train_multi(X, y, lr, n_epochs):
    ...
```

Funkcija `make_data` ustvari podatke z resničnimi utežmi `w = [2, 3, -1]` in
prostim členom `b = 1`.

## Vrnjena vrednost

- `train_multi(X, y, lr, n_epochs)` naj vrne `(w, b)`, kjer je `w` vektor uteži
  (`numpy` polje) in `b` prosti člen. Začni pri ničlah.

## Raziščite sami

Ustvari več podatkovnih množic z enako matriko `X` in različnimi vrednostmi
standardnega odklona šuma (npr. `0, 1, 5, 20`). Za vsako množico nauči model in
primerjaj ocenjene uteži ter prosti člen s pravimi vrednostmi
(`w = [2, 3, -1]`, `b = 1`). Kako natančno model pri različnih ravneh šuma oceni
prave vrednosti uteži?

```python
for noise in [0.0, 1.0, 5.0, 20.0]:
    X, y = make_data(noise=noise)
    w, b = train_multi(X, y, lr=0.05, n_epochs=3000)
    print(f"noise={noise:<5} w={np.round(w, 3)} b={b:.3f}")
```

## Testiranje

```bash
python -m unittest -v test_lin_reg
```
