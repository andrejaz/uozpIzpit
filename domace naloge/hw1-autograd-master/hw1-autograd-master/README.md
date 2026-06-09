V nalogi boste s sistemom za avtomatsko odvajanje implementirali funkciji `fn1(x1, x2)` in `fn2(x1, x2)`, ki vračata objekt tipa `Value` in izračunata:

```math
\mathrm{fn}_1 = \frac{1}{1-z},\quad \text{kjer je } z = 0.2 + 0.5x_1 + x_2
```

```math
\mathrm{fn}_2 = \mathrm{ln}(\frac{1}{1+e^{-z}}),\quad \text{kjer je } z = 0.2 + 0.5x_1 + x_2 + \mathrm{ln}x_2
```

Da ju boste lahko implementirali, razred `Value` s predavanja razširite s potrebno funkcionalnostjo za izračun in odvajanje zgornjih funkcij.

Rešitev pred oddajo lokalno preverite s `python -m unittest test_autograd.py`