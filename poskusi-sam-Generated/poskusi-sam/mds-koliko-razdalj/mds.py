import numpy as np


DISTANCES = {
    ("Novo Mesto", "Maribor"): 170,
    ("Novo Mesto", "Celje"): 83,
    ("Novo Mesto", "Koper"): 169,
    ("Novo Mesto", "Kranj"): 99,
    ("Novo Mesto", "Ljubljana"): 72,
    ("Novo Mesto", "Postojna"): 116,
    ("Maribor", "Celje"): 55,
    ("Maribor", "Koper"): 232,
    ("Maribor", "Kranj"): 156,
    ("Maribor", "Ljubljana"): 128,
    ("Maribor", "Postojna"): 178,
    ("Celje", "Koper"): 183,
    ("Celje", "Kranj"): 105,
    ("Celje", "Ljubljana"): 77,
    ("Celje", "Postojna"): 130,
    ("Koper", "Kranj"): 128,
    ("Koper", "Ljubljana"): 107,
    ("Koper", "Postojna"): 58,
    ("Kranj", "Ljubljana"): 30,
    ("Kranj", "Postojna"): 77,
    ("Ljubljana", "Postojna"): 53,
}


def stress(positions, distances):
    """Vsota kvadratov razlik (||pos_a - pos_b|| - razdalja)^2 čez pare."""
    raise NotImplementedError


def mds(distances, dim=2, lr=0.01, epochs=2000, seed=0):
    """Z gradientnim sestopom minimizira stress. Vrne slovar mesto -> položaj."""
    raise NotImplementedError


if __name__ == "__main__":
    pos = mds(DISTANCES)
    print("koncni stress:", round(stress(pos, DISTANCES), 2))
