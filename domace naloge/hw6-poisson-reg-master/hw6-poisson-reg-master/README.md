V datoteki `data.csv` so shranjeni podatki s celoštevilsko ciljno spremenljivko.

V nalogi implementirajte linearno in Poissonovo regresijo ter ju aplicirajte na podane podatke. Implementacijo linearne regresije (in pripadajočo funkcijo za optimizacijo, `train`) lahko vzamete iz prejšnjih predavanj. Poissonovo regresijo bo najlaže implementirati kot podrazred linearne regresije; pri tem bo treba spremeniti vsaj metodi `__call__`, ki računa napovedi, ter `loss`, ki definira optimizacijsko funkcijo.

Ste opazili, kako močno lahko predpostavka ciljne porazdelitve vpliva na kakovost napovedi?

Rešitev naloge je kratka in preprosta. Toplo priporočamo, da pri reševanju **kot literaturo uporabljate izključno zapiske s predavanj**.

Pri implementaciji lahko uporabite samo standardne Pythonove knjižnice ter knjižnici `pandas` in `numpy`. Implementacijo shranite v datoteko `poisson.py`. Rešitev preverite z ukazom:

```
python -m unittest -v test_poisson
```