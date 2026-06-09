def first_principal_component(X):
    """Enotni smerni vektor prve glavne komponente."""
    #centriranje podatkov
    X_centered = X - np.mean(X, axis=0)

    #kovariancna matrika 
    C = np.cov(X_centered, rowvar=False)

    #lastne vrednosti in lastni vektroji
    w, v = np.linalg.eigh(C)

    return v[:, -1]
