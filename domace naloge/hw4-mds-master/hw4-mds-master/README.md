V datoteki `razdalje.yaml` so podani potovalni časi vlakov med pari mest (vemo, da niso ravno kratki). Uredite mesta na časovno premico, pri čemer uporabite večrazredno lestvičenje. Z drugimi besedami: implementirajte MDS (Multidimensional Scaling), tako da podatke (mesta) razporedite na časovno premico (enodimenzionalni MDS), pri čemer naj bodo pari razdalj med mesti na tej premici čim bolj podobni parom razdalj v vhodnih podatkih.

Za cenilno funkcijo MDS uporabite naslednjo formulo:

```math
J = \frac{\sum_{(i,j) \in D}(d_{ij}-\|x_i - x_j\|_2)^2}{|D|}
```

Implementirajte tri variante naloge:

1. Enorazsežni MDS (`MDS1D`); ugotovili boste, da za podane podatke in inicializacijo običajno ne deluje dobro.
2. Dvorazsežni MDS (`MDS2D`); ta deluje odlično, vendar pri podanih podatkih dobimo eno odvečno dimenzijo.
3. Dvorazsežni MDS z regularizacijo druge dimenzije (`MDS2DReg`), s katerim efektivno dobimo enorazsežni MDS.

Začetne vrednosti rešitev naj bodo vedno uniformno naključno izbrane na intervalu [-100, 100].

Za del naloge, ki ga oddate na GitHub, lahko uporabite samo knjižnice za branje datotek (yaml) in knjižnico za strojno odvajanje, ki ste jo razvili na predavanjih in uporabljali pri prvih dveh domačih nalogah.

Tokrat na **spletno učilnico oddajte PDF** z grafično izrisanimi rezultati za vse tri variante implementacije MDS. Za izris uporabite knjižnico `matplotlib`.

Rezultat naj bo slovar mest z njihovimi pozicijami na časovni premici. Implementacijo shranite v eno samo datoteko `mds-1d.py`. Rešitev preverite z ukazom:

```
python -m unittest -v test_mds_1d
```