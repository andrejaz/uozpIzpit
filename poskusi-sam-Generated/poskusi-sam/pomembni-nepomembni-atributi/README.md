# Pomembni in nepomembni atributi

## Opis

Modelu dodaj nepomembne (naključne) atribute in preveri, ali jih linearna
regresija pravilno prepozna kot manj pomembne.

## Naloga

V datoteki `lin_reg.py` dopolni funkcijo:

```python
def rank_features(w):
    ...
```

Funkciji `make_data` in `train_multi` sta že podani.

## Vrnjena vrednost

- `rank_features(w)` naj vrne indekse atributov, urejene padajoče po absolutni
  vrednosti uteži.

## Raziščite sami

Ustvari podatke, pri katerih je `y` odvisen le od prvih treh atributov, in dodaj
pet naključnih atributov, ki niso povezani z `y`. Nauči model
`train_multi` in atribute razvrsti po absolutni vrednosti ocenjenih uteži.
Preveri, ali so med tremi najvišje uvrščenimi res prvi trije atributi. Kaj
opaziš pri ostalih?

## Testiranje

```bash
python -m unittest -v test_lin_reg
```
