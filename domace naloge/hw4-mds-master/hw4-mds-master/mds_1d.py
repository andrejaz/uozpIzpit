import random
import yaml

from value import Value


class MDS1D:
    def __init__(self, items):
        self.pos = {i: Value(random.uniform(-100, 100)) for i in items}

    def __call__(self, x):
        # vrne koordinato mesta x
        return self.pos[x]

    def parameters(self):
        return list(self.pos.values())

    def loss(self, data):
        total_loss = Value(0)
        for (mesto1, mesto2), real_dist in data.items():
            pos1 = self(mesto1)
            pos2 = self(mesto2)

            distance_model = (pos1 - pos2)**2
            razlika = (distance_model**0.5 - real_dist)**2

            total_loss += razlika
        return total_loss / len(data)


def flatten1d(l):
    if not len(l):
        return []
    rest = flatten1d(l[1:])
    this = list(l[0]) if isinstance(l[0], (list, tuple)) else [l[0]]
    return this + rest


class MDS2D:
    def __init__(self, items):
        self.pos = {i: (Value(random.uniform(-100, 100)), Value(random.uniform(-100, 100))) for i in items}

    def __call__(self, x):
        # vrne koordinato mesta x
        return self.pos[x]

    def parameters(self):
        return flatten1d(list(self.pos.values()))

    def loss(self, data):
        total_loss = Value(0)
        for (mesto1, mesto2), d_ij in data.items():
            x1, y1 = self.pos[mesto1]
            x2, y2 = self.pos[mesto2]
            # Evklidska razdalja v 2D
            dist_sq = (x1 + (-1 * x2))**2 + (y1 + (-1 * y2))**2
            pred_dist = dist_sq**0.5
            total_loss += (d_ij + (-1 * pred_dist))**2
        return total_loss / len(data)
    


class MDS2DReg(MDS2D):
    # Dodajte regularizacijo po 2. komponenti

    def loss(self, data):
        base_loss = super().loss(data)
        reg_penalty = Value(0)
        for _, y in self.pos.values():
            reg_penalty += y**2
        return base_loss + (reg_penalty / len(self.pos))


def train(model, distances, learning_rate=1, n_epochs=500):
    for k in range(n_epochs):
        loss = model.loss(distances)
        loss.backward()

        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if k % 100 == 0:
            print(f"{k:3} Loss: {model.loss(distances).data:5.3f}")
    return model


def read_distances():
    with open('razdalje.yaml', 'r', encoding='utf8') as file:
        distances = yaml.safe_load(file)
    distances = {tuple(k.split(' -> ')): v for k, v in distances.items()}
    return distances


if __name__ == "__main__":
    # load data from razdalje.yaml
    distances = read_distances()
    items = set(i for pair in distances.keys() for i in pair)

    random.seed(0)
    model = MDS2D(items)
    model = train(model, distances)

    # Printout

    def get_values(item):
        return [a.data for a in item] if isinstance(item, (tuple, list)) else item.data

    city_positions = {city: get_values(model(city)) for city in items}
    sorted_cities = sorted(city_positions.items(), key=lambda x: x[1])

    # Print cities in order
    print("\nCities in order (from left to right):")
    for city, pos in sorted_cities:
        print(f"{city}: {pos}")