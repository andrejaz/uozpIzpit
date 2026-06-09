def polynomial_features(x, degree):
    """Za vektor x vrne matriko stolpcev [x, x^2, ..., x^degree]."""
    n = len(x)
    m = []
    
    for i in range(n):
        vrstica = []
        for d in range(1, degree + 1):
            vrstica.append(x[i]**d)
        m.append(vrstica)
    
    # Pretvorimo v numpy matriko, da bo delovalo z matrikami
    return np.array(m)