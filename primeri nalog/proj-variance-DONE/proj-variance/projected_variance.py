import math


def projected_variance(X, direction):
    """
    Vrne varianco projekcij podatkov X na dano smer.

    Podatki v X so centrirani. Smerni vektor direction ni nujno enotski.
    """
    # TODO: normirajte smerni vektor na enotsko dolžino
    vsota = sum(x**2 for x in direction)
    dolzina = math.sqrt(vsota)
    x_norm = [x/dolzina for x in direction]
    # TODO: izračunajte projekcije vseh točk in vrnite povprečje njihovih kvadratov
    #projekcija tocke na smer: pi = x1 dot ux + x2 dot uy
    pi = [(x[0] * x_norm[0] + x[1] * x_norm[1])**2  for x in X]

    var = (sum(pi)) / len(X)
    return var




if __name__ == "__main__":
    # Preprost primer za ročno testiranje
    X = [(2.0, 0.0), (-2.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
    result = projected_variance(X, (-1.0, 1.0))
    print(round(result, 4))
