# Kdaj glavna komponenta sledi šumu?

## Opis

Prva glavna komponenta kaže v smer največje variance. Če je struktura podatkov
(npr. ločitev dveh skupin) v vodoravni smeri, navpični šum pa narašča, lahko
glavna komponenta v nekem trenutku preneha slediti ločitvi in začne slediti
šumu.

## Naloga

V datoteki `pca.py` dopolni funkcijo:

```python
def first_principal_component(X):
    ...
```

Funkcija `make_two_groups(noise, ...)` (dve vodoravno ločeni skupini z navpičnim
šumom) je že podana.

## Vrnjena vrednost

- `first_principal_component(X)` naj vrne enotni smerni vektor prve glavne
  komponente (lastni vektor kovariančne matrike z največjo lastno vrednostjo).

## Raziščite sami

Ustvari dve skupini točk, ki sta na začetku dobro ločeni v vodoravni smeri, nato
postopoma povečuj navpični šum. Za vsako stopnjo šuma izračunaj prvo glavno
komponento in jo primerjaj z intuitivno smerjo ločevanja skupin. Opazuj, pri
kateri stopnji šuma prva komponenta preneha slediti ločevanju in začne slediti
šumu.

```python
for noise in [0.1, 1.0, 3.0, 6.0, 10.0]:
    v = first_principal_component(make_two_groups(noise=noise, seed=0))
    print(f"noise={noise:<5} smer=({v[0]:+.2f}, {v[1]:+.2f})")
```

## Testiranje

```bash
python -m unittest -v test_pca
```
