# Razlaga gruč na podatkih o prekinitvi dela

## Opis

Zaposlene razvrsti v gruče in izbrano gručo razloži z značilkami, ki jo najbolj
ločijo od preostalih. Za rangiranje uporabi Mann–Whitneyjevo statistiko (AUC),
ki pove, kako pogosto je vrednost značilke v gruči uvrščena višje (tu: nižja
vrednost = višje uvrščena) kot zunaj nje.

## Naloga

V datoteki `attrition.py` dopolni funkcijo:

```python
def mann_whitney_auc(group, other):
    ...
```

Funkciji `kmeans` in `load_data` (prebere `attrition.csv`) sta podani.

## Podatki

Datoteko `attrition.csv` po potrebi ustvariš z:

```bash
python generate_data.py
```

## Vrnjena vrednost

- `mann_whitney_auc(group, other)` naj vrne delež parov `(g, o)`, pri katerih je
  vrednost v gruči manjša od vrednosti zunaj gruče (vezi štejejo pol):
  `AUC = (#(g < o) + 0.5 * #(g == o)) / (|group| * |other|)`. Vrednost blizu 1
  pomeni, da so vrednosti gruče skoncentrirane pri vrhu (nizke vrednosti).

## Raziščite sami

Zaposlene razvrsti v gruče (npr. z metodo voditeljev, brez ciljne spremenljivke
Attrition). Izberi gručo in z domenskim znanjem oblikuj smiselne skupine značilk
(npr. zadovoljstvo, plače, obremenitve). Za vsako izračunaj obogatenost po
Mann–Whitneyjevi statistiki. Kateri domenski vidiki najbolje opisujejo izbrano
gručo?

## Testiranje

```bash
python -m unittest -v test_attrition
```
