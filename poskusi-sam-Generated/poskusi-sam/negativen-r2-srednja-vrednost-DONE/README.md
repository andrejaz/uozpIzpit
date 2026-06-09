# Negativen R² pri napovedih s srednjo vrednostjo

## Opis

Če na testni množici napoveduješ s povprečjem ciljne spremenljivke **učne**
množice, je pričakovana vrednost `R²` enaka 0. V praksi pa lahko dobiš tudi
negativno vrednost.

## Naloga

V datoteki `r2.py` dopolni funkciji:

```python
def r2_score(y_true, y_pred):
    ...


def baseline_predict(y_train, n_test):
    ...
```

## Vrnjena vrednost

- `r2_score(y_true, y_pred)` naj vrne `1 - SS_res / SS_tot`.
- `baseline_predict(y_train, n_test)` naj vrne polje dolžine `n_test`, napolnjeno
  s povprečjem `y_train`.

## Raziščite sami

Podobno kot prej, le da napoveduješ s srednjo vrednostjo odvisne spremenljivke
učne množice. Pričakovana vrednost `R²` na testni množici je 0. Kaj v resnici
dobiš? Preveri, kako je to odvisno od velikosti učnih množic.

## Testiranje

```bash
python -m unittest -v test_r2
```
