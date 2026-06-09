# Kakšna je vloga parametra σ? (t-SNE)

## Opis

V t-SNE parameter σ določa širino Gaussovega jedra, ki podobnosti med pari
primerov v izvornem prostoru pretvori v verjetnosti `p_{j|i}`. Majhen σ poudari
zelo lokalne sosede, velik σ pa se približa globalni strukturi.

## Naloga

V datoteki `tsne.py` dopolni funkciji:

```python
def gaussian_affinities(D, sigma):
    ...


def make_clusters(n_per, sep, seed=0):
    ...
```

Za raziskovanje je na voljo poenostavljen `tsne` v `tsne_simple.py`.

## Vrnjena vrednost

- `gaussian_affinities(D, sigma)` naj iz matrike razdalj `D` izračuna pogojne
  verjetnosti: `p_{j|i} ∝ exp(-D_{ij}^2 / (2σ^2))` za `i ≠ j`, diagonala je 0,
  vsaka vrstica pa je normirana, da je vsota verjetnosti enaka 1.
- `make_clusters(n_per, sep, seed)` naj ustvari tri 2D skupine, ločene z razmikom
  `sep`, in vrne `(X, labels)`.

## Raziščite sami

Na podatkih z več jasno ločenimi skupinami izvedi t-SNE za različne vrednosti σ.
Opazuj, kako se spreminja oblika vložitve. Pri majhnih vrednostih metoda
poudarja lokalne sosede, pri velikih pa se približuje globalni strukturi. Poskusi
najti vrednost, pri kateri so skupine najbolj jasno ločene.

## Testiranje

```bash
python -m unittest -v test_tsne
```
