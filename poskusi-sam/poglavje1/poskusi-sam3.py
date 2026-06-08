def f(a,b,c,d):
    return (a*b + c)*d

def parc_a(a,b,c,d, h=0.00001):
    return (f(a+h, b,c,d) - f(a,b,c,d)) / h

def parc_b(a,b,c,d, h=0.00001):
    return (f(a, b+h,c,d) - f(a,b,c,d)) / h


def parc_c(a,b,c,d,h=0.00001):
    return (f(a,b,c+h,d) - f(a,b,c,d)) / h


def parc_d(a,b,c,d,h=0.00001):
    return (f(a,b,c,d+h) - f(a,b,c,d)) / h


a = 2
b = -3
c = 10
d = -2
print(f"po a: {parc_a(a,b,c,d)}\npo b: {parc_b(a,b,c,d)}\npo c:{parc_c(a,b,c,d)}\npo d:{parc_d(a,b,c,d)}")
