# Vzvratni prehod za tanh

## Opis

Razširi metodo `tanh` v razredu `Value` z implementacijo `_backward()`. Odvod
funkcije tanh je `1 - tanh^2(x)`, zato za izhodno vozlišče `out = self.tanh()`
velja:

```text
self.grad += (1 - out.data**2) * out.grad
```

## Naloga

V datoteki `value_tanh_backward.py` dopolni metodo `tanh` tako, da poleg
vrednosti pravilno nastavi tudi `_backward`.

```python
def tanh(self):
    ...
```

## Raziščite sami

Zgradi graf za izraz `(a + b).tanh() * c` z vrednostmi `a = 1.0`, `b = 0.5`,
`c = 2.0`, pokliči `backward()` in izpiši gradiente po `a`, `b` in `c`. Nato z
metodo končnih diferenc numerično oceni iste parcialne odvode in primerjaj
rezultate; če se razlikujejo, poišči napako v `forward` ali `backward` delu.

```python
a, b, c = Value(1.0), Value(0.5), Value(2.0)
out = (a + b).tanh() * c
out.backward()
print(a.grad, b.grad, c.grad)
```

## Testiranje

```bash
python -m unittest -v test_value_tanh_backward
```
