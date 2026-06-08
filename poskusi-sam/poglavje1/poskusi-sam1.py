def f(a):
    return a**2 - 10 * a + 28

def odvod(a, h):
    return (f(a+h)-f(a))/h

def analiticniOdvod(a):
    return 2*a - 10

a = 6
h1 = 10**(-1)
h2 = 10**(-2)
h3 = 10**(-4)
h4 = 10**(-8)

print(f"h1: {h1}")
print(f"h1: {h2}")
print(f"h1: {h3}")
print(f"h1: {h4}")



print("-----rezultati----")
print(f"h1, analiticni: {odvod(a, h1)}, {analiticniOdvod(a)}")
print(f"h2, analiticni: {odvod(a, h2)}, {analiticniOdvod(a)}")
print(f"h3, analiticni: {odvod(a, h3)}, {analiticniOdvod(a)}")
print(f"h4, analiticni: {odvod(a, h4)}, {analiticniOdvod(a)}")

