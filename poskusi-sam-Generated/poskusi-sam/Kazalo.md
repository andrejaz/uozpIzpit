# Poskusi sam

Naloge »Poskusi sam« iz učbenika *Uvod v odkrivanje znanj iz podatkov* (Blaž
Zupan, 2026). Vsaka podmapa vsebuje eno nalogo s tremi (ali več) datotekami:

- `README.md` – opis naloge, kaj je treba implementirati in kako jo raziskati;
- okostje (`*.py`) – funkcije z dokumentacijo, ki jih je treba dopolniti (mesta
  označena z `raise NotImplementedError`);
- `test_*.py` – testi enot za preverjanje rešitve;
- po potrebi `generate_data.py` in pripadajoča podatkovna datoteka (`*.csv`).

## Kako reševati

V mapi naloge dopolni okostje, nato poženi teste:

```bash
cd <mapa-naloge>
python -m unittest -v
```

Dovoljeni knjižnici sta `numpy` (povsod) in `torch` ter `scipy` (kjer je to v
nalogi navedeno). Za vizualizacije iz razdelkov »Raziščite sami« lahko uporabiš
`matplotlib`.

## Kazalo nalog

### Gradientni sestop, odvajanje in računski graf
- [koncne-diference-korak](koncne-diference-korak/) – Korak pri končnih diferencah
- [gradientni-spust-1d](gradientni-spust-1d/) – Gradientni spust v eni spremenljivki
- [gradient-koncne-diference](gradient-koncne-diference/) – Gradient s končnimi diferencami

### Strojno odvajanje
- [value-tanh](value-tanh/) – Operacija tanh v razredu Value
- [value-tanh-backward](value-tanh-backward/) – Vzvratni prehod za tanh
- [gradientni-sestop-2d](gradientni-sestop-2d/) – Gradientni sestop v dveh spremenljivkah

### Linearna regresija
- [stopnja-ucenja-konvergenca](stopnja-ucenja-konvergenca/) – Stopnja učenja in konvergenca
- [pot-gradientnega-sestopa](pot-gradientnega-sestopa/) – Pot gradientnega sestopa
- [kriticna-stopnja-ucenja](kriticna-stopnja-ucenja/) – Kritična stopnja učenja
- [velikost-paketa-konvergenca](velikost-paketa-konvergenca/) – Velikost paketa in konvergenca
- [pot-paketnega-sestopa](pot-paketnega-sestopa/) – Pot paketnega sestopa
- [vpliv-suma-utezi](vpliv-suma-utezi/) – Vpliv šuma na oceno uteži
- [pomembni-nepomembni-atributi](pomembni-nepomembni-atributi/) – Pomembni in nepomembni atributi

### Regularizacija in izbor značilk
- [negativen-r2-test](negativen-r2-test/) – Negativen R² na testni množici
- [negativen-r2-srednja-vrednost](negativen-r2-srednja-vrednost/) – Negativen R² pri napovedih s srednjo vrednostjo
- [regularizacijska-pot](regularizacijska-pot/) – Regularizacijska pot in nestabilnost izbora značilk
- [korelirane-znacilke](korelirane-znacilke/) – Korelirane značilke
- [uhajanje-informacij](uhajanje-informacij/) – Uhajanje informacij pri standardizaciji in izboru λ

### Glavne komponente (PCA)
- [pca-sledi-sumu](pca-sledi-sumu/) – Kdaj glavna komponenta sledi šumu
- [pca-krog](pca-krog/) – Zakaj PCA ne vidi kroga
- [pca-nakljucni-podatki](pca-nakljucni-podatki/) – Koliko variance pojasni prva komponenta pri naključnih podatkih
- [pca-skupine-atributov](pca-skupine-atributov/) – Ali PCA odkrije skupine povezanih atributov
- [pca-pojasnjena-varianca-test](pca-pojasnjena-varianca-test/) – Pojasnjena varianca na testni množici
- [pca-analiticni](pca-analiticni/) – Analitični PCA in gradientni sestop

### Dvodimenzionalne vložitve podatkov
- [mds-koliko-razdalj](mds-koliko-razdalj/) – Koliko razdalj potrebujemo za zemljevid
- [mds-odpornost-na-sum](mds-odpornost-na-sum/) – Odpornost vložitev na šum
- [mds-tsne-primerjava](mds-tsne-primerjava/) – Kaj vidi posamezna metoda (MDS in t-SNE)
- [tsne-parameter-sigma](tsne-parameter-sigma/) – Kakšna je vloga parametra σ

### Gručenje in razlaga gruč
- [veckratno-testiranje](veckratno-testiranje/) – Preveč skupin, premalo statistike
- [izmisli-si-gruco](izmisli-si-gruco/) – Izmisli si gručo
- [razlaga-gruc-attrition](razlaga-gruc-attrition/) – Razlaga gruč na podatkih o prekinitvi dela
- [silhueta-lov](silhueta-lov/) – Lov na največjo silhueto
- [voditelji-gradientni-sestop](voditelji-gradientni-sestop/) – Metoda voditeljev z gradientnim sestopom
- [gruce-drzav-mreza](gruce-drzav-mreza/) – Gruče držav na mreži podobnosti

## Opomba o podatkih

Naloge, ki v učbeniku uporabljajo zunanje podatke (Kaggle/UCI), tu uporabljajo
**sintetične** nadomestne množice, ustvarjene s skriptami `generate_data.py`, da
so neposredno izvedljive. Po želji jih zamenjaš z izvirnimi podatki.
