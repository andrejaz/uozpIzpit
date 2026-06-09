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
    rez = 0

    for (mesto1, mesto2), razdalja in distances.items():
        pos_a = positions[mesto1]
        pos_b = positions[mesto2]

        izracunana_razdalja = np.linalg.norm(pos_a- pos_b)
        razlika = np.abs(razdalja - izracunana_razdalja)
        razlika = razlika ** 2
        rez += razlika

    return rez 



def mds(distances, dim=2, lr=0.01, epochs=2000, seed=0):
    """Z gradientnim sestopom minimizira stress. Vrne slovar mesto -> položaj."""
    rng = np.random.default_rng(seed)
    mesta = set()
    for mesto1, mesto2 in distances.keys():
        mesta.add(mesto1)
        mesta.add(mesto2)

    mesta_pozicije = {}

    for mesto in mesta:
        mesta_pozicije[mesto] = rng.random(dim)

    
    for i in range(epochs):
        grads = {m: np.zeros(dim) for m in mesta}
        
        for (m1, m2), d_real in distances.items():
            pos1 = mesta_pozicije[m1]
            pos2 = mesta_pozicije[m2]
            
            d_map = np.linalg.norm(pos1 - pos2)
            
            if d_map < 1e-9:
                continue
            
            factor = 2 * (d_map - d_real) / d_map
            
            grads[m1] += factor * (pos1 - pos2)
            grads[m2] += factor * (pos2 - pos1)

        for m in mesta:
            mesta_pozicije[m] -= lr * grads[m]
    return mesta_pozicije


if __name__ == "__main__":
    pos = mds(DISTANCES)
    print("koncni stress:", round(stress(pos, DISTANCES), 2))
