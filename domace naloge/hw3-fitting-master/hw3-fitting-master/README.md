V nalogi boste za linearno regresijo in podatke `body-fat-brozek.csv` s prečnim preverjanjem poiskali ustrezno stopnjo regularizacije. To je mogoče implementirati na več napačnih načinov. Pravilni pristopi (le-teh je manj) pa imajo naslednji lastnosti:

- **Ne testiramo na učnih podatkih!** Modelov nikoli ne ocenjujemo na podatkovni množici, ki je bila uporabljena za učenje. Učenje tukaj vključuje tako optimizacijo uteži kot tudi iskanje regularizacijskega parametra (v splošnem pa vse, kar vpliva na končni model).
- **Vedno izkoristimo vse podatke!** Ko izberemo ustrezne parametre modela (in model ocenimo), končni model (tj. model za produkcijo) z izbranimi parametri še enkrat naučimo na vseh podatkih.

V nalogi boste uporabili linearno regresijo z L1 in L2 regularizacijo. Ker je poudarek naloge na primernem izboru stopnje regularizacije in ne na samem modelu, boste uporabili robustne (in hitrejše) implementacije linearne regresije iz knjižnice `scikit-learn`: razreda `Ridge` in `Lasso`. Razen teh razredov ne smete uporabljati **ničesar drugega** iz knjižnice `scikit-learn`.

Cilj naloge je, da sami implementirate analogno funkcionalnost knjižnice `scikit-learn` za iskanje stopnje regularizacije (tukaj jo označujemo z `alpha`; na predavanjih je bila označena z `lambda`). V knjižnici `scikit-learn` to poteka na zelo pregleden način: skozi nov model `GridSearchCV`, ki zaobjame podanega. Tako lepo enkapsulira kompleksnost iskanja, zato nam o dvojnem prečnem preverjanju sploh ni treba razmišljati.  Implementirati bo treba torej naslednjo funkcionalnost:

```
# model, ki standardizira podatke
pipe = make_pipeline(StandardScaler(), Ridge())
# model, ki z notranjim prečnim preverjanjem izbere stopnjo regularizacije prejšnjega
cvm = GridSearchCV(estimator=pipe, param_grid={"ridge__alpha": alpha_fit},
                   scoring="r2", cv=CVSplitter(5))

# evaluacija modela
s_score = np.mean(cross_val_score(cvm, X, ys, scoring="r2", cv=CVSplitter(5)))

# gradnja končnega modela za produkcijo
cvm.fit(X, ys)
s_best_alpha = cvm.best_params_["ridge__alpha"]  # najboljši parameter
s_pred5 = cvm.predict(X[:5])
```

Pri reševanju postopajte takole:

1. Implementirajte razred `LinearStandardizedModel` s funkcijama `fit` in `predict`, ki delujeta kot v `scikit-learn`. Razred naj podpira tako L1 in L2 regularizacijo s poljubno stopnjo. Standardizacijo (na predavanjih imenovano tudi normalizacija) naj izvaja avtomatsko oz. interno.
2. Oglejte si implementacijo `CVSplitter`, ki zgradi indekse v učno in testno množico za prečno preverjanje. Razred `CVSplitter` je definiran tako, da zagotavlja enake rezultate pri uporabi knjižnice `scikit-learn` in vaše implementacije.
3. Implementirajte funkcijo `cross_val_r2`, ki podan model oceni s prečnim preverjanjem in oceno R2.
4. Implementirajte razred `FittedLinearStandardizedModel`, ki z notranjim prečnim preverjanjem izbere ustrezno stopnjo regularizacije (in z njo ponovno nauči model na vseh podatkih).

Pri tem si pomagajte tudi z dokumentacijo in kodo `scikit-learn`. Testi bodo preverili, ali vaša implementacija vrača popolnoma enake rezultate.

Testi (in predloga) natančneje definirajo vmesnik funkcij in razredov, ki jih je treba implementirati. Še enkrat: iz knjižnice `scikit-learn` lahko uporabite zgolj razreda `Ridge` in `Lasso`, **ničesar drugega**.

Rešitev pred oddajo lokalno preverite s `python -m unittest test_fitting`. Čas izvajanja naloge na strežniku je omejen na 1 minuto.