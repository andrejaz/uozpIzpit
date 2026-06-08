# Operacija tanh v razredu Value

## Opis

Razred `Value` predstavlja vozlišče računskega grafa za strojno odvajanje.
V tej nalogi dodaj metodo `tanh`, ki na enak način kot `__add__` in `__mul__`
ustvari novo vozlišče z vrednostjo `tanh(self.data)`, si zapomni predhodnika in
oznako operacije (`"tanh"`). Vzvratnega prehoda (`_backward`) tu še ne
implementiramo – to je naloga `value-tanh-backward`.

## Naloga

V datoteki `value_tanh.py` dopolni metodo:

```python
def tanh(self):
    ...
```

(Namig: uporabi `math.tanh`.)

## Vrnjena vrednost

- `tanh()` naj vrne novo vozlišče `Value` z vrednostjo `math.tanh(self.data)`,
  med predhodnike (`_children`) vključi `self`, oznaka operacije (`_op`) naj bo
  `"tanh"`.

## Raziščite sami

Zgradi graf za izraz `(a + b).tanh() * c` z vrednostmi `a = 1.0`, `b = 0.5`,
`c = 2.0`, izpiši končno vrednost in preveri, da ima izhodno vozlišče pravilne
predhodnike v `_prev`.

```python
a = Value(1.0, label="a")
b = Value(0.5, label="b")
c = Value(2.0, label="c")
out = (a + b).tanh() * c
print(out.data)
print(out._prev)
```

## Testiranje

```bash
python -m unittest -v test_value_tanh
```
