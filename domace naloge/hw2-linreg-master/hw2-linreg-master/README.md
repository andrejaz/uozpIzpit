V nalogi uporabite linearno regresijo s predavanj na podatkih `body-fat-brozek.csv`. Naloga od vas zahteva uporabo linearne regresije (kar morate tudi oddati), poleg tega pa vam v razmislek ponuja nekaj vprašanj (odgovorov ne rabite oddati).

Napišite funkcije, ki preberejo podatke ter zgradijo različne modele z linearno regresijo:

1. Model zgrajen na vseh atributih (funkcija `model_all`).
2. Model zgrajen samo na atributih o teži in višini (funkcija `model_wh`).
3. Model zgrajen samo na atributih o teži in višini ter na njunih kvadratih (skupno 4 atributi, normalizacijo izvedite po kvadriranju, funkcija `model_wh_squared`).

Vse funkcije morajo vrniti zgrajen model ter podatke, na katerih je bil model zgrajen. Uporabljajte enako vrsto normalizacije kot na predavanjih. Točnejšo specifikacijo lahko razberete iz testov.

Za vsakega od zgornjih modelov si izpišite še uteži atributov (tega testi sicer ne preverjajo). Zakaj se model (3) izkaže za boljšega kot model (2)?

Naredite še en poskus: model (3) razširite tako, da pred učenjem odstrani poljubno vrstico (funkcija `model_wh_squared_remove_row`). Poglejte, kako se model spremeni, če odstranite učni primer z indeksom 41, in kako, če odstranite drug primer. Zakaj je takšna razlika?

Pred implementacijo zaradi testov najprej zagotovite, da metoda `loss` razreda `LinReg` vrača povprečno kvadratno napako za vsak primer v učni množici, ne pa njihovega seštevka (to preveri `test_loss_mean`); vsaj prva verzija iz zapiskov s predavanj je delovala malenkost drugače.

Rešitev pred oddajo lokalno preverite s `python -m unittest test_bodyfat`. Čas izvajanja naloge na strežniku je omejen na 2 minuti.