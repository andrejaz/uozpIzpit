def f(a):
    """Testna funkcija f(a) = a^2 - 10a + 28."""
    return a ** 2 - 10 * a + 28


def forward_difference(func, a, h):
    """Oceni odvod funkcije func v točki a z metodo končnih diferenc:
    (func(a + h) - func(a)) / h."""
    return (func(a+h)-func(a))/h


def analytic_derivative(a):
    """Vrne analitični odvod funkcije f v točki a (f'(a) = 2a - 10)."""
    return 2*a - 10


if __name__ == "__main__":
    for h in [1e-1, 1e-2, 1e-4, 1e-8, 1e-12]:
        approx = forward_difference(f, 6.0, h)
        print(f"h={h:.0e}  ocena={approx:.8f}  napaka={abs(approx - 2.0):.2e}")
