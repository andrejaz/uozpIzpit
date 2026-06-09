# Korelirane značilke

## Opis

Pri močno koreliranih atributih L1 (LASSO) regularizacija izbere le nekatere od
njih, izbor pa je lahko nestabilen (odvisen od začetnih uteži).

## Naloga

V datoteki `korelacije.py` dopolni funkcijo:

```python
def selected_features(w, tol=1e-8):
    ...
```

Funkciji `fit` (L1/L2 regularizirana regresija) in `make_correlated_data` sta že
podani. `make_correlated_data` ustvari množico, kjer je cilj odvisen od treh
atributov, dodani pa so še atributi, močno korelirani s prvim.

## Vrnjena vrednost

- `selected_features(w, tol)` naj vrne množico indeksov atributov z neničelno
  utežjo (`|w| > tol`).

## Raziščite sami

Nauči model z L1 regularizacijo pri izbrani stopnji regularizacije `λ`. Učenje
ponovi večkrat z različnimi naključnimi začetnimi utežmi (`seed`). Preveri, ali
LASSO med koreliranimi značilkami vedno izbere isto. Kaj ti ugotovitve povedo o
razlagi modela z ničelnimi utežmi?

```python
X, y = make_correlated_data()
for seed in range(5):
    w, b = fit(X, y, reg="l1", reg_strength=0.2, seed=seed)
    print(sorted(selected_features(w)))
```

## Testiranje

```bash
python -m unittest -v test_korelacije
```
