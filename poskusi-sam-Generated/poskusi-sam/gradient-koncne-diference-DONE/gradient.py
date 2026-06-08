def L(a, b, c, d):
    """Funkcija L(a, b, c, d) = (a*b + c) * d."""
    return (a * b + c) * d


def numerical_gradient(func, params, h=1e-5):
    """Oceni gradient func glede na vsak parameter v params (a, b, c, d).
    Vrne seznam parcialnih odvodov."""
    a = params[0]
    b = params[1]
    c = params[2]
    d = params[3]
    parc_a = (func(a+h, b,c,d)-func(a,b,c,d))/h
    parc_b = (func(a, b+h,c,d)-func(a,b,c,d))/h
    parc_c = (func(a, b,c+h,d)-func(a,b,c,d))/h
    parc_d = (func(a, b,c,d+h)-func(a,b,c,d))/h
    #return [int(parc_a), int(parc_b), int(parc_c), int(parc_d)]
    return [parc_a, parc_b, parc_c, parc_d]


def analytic_gradient(a, b, c, d):
    """Vrne (dL/da, dL/db, dL/dc, dL/dd) = (b*d, a*d, d, a*b + c)."""
    return [b*d, a*d, d, a*b+c]


if __name__ == "__main__":
    print("analiticni:", analytic_gradient(2, 3, 10, 2))
    print("numericni: ", numerical_gradient(L, (2, 3, 10, 2)))
