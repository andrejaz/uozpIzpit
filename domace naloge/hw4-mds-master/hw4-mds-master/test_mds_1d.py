import unittest
import yaml

from mds_1d import MDS1D, MDS2D, MDS2DReg


def read_distances():
    with open('razdalje.yaml', 'r', encoding='utf8') as file:
        distances = yaml.safe_load(file)
    distances = {tuple(k.split(' -> ')): v for k, v in distances.items()}
    return distances


def point_values(item):
    return [a.data for a in item] if isinstance(item, (tuple, list)) else [item.data]


def train(model, distances, learning_rate=1, n_epochs=500):
    for k in range(n_epochs):
        loss = model.loss(distances)
        loss.backward()

        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if k % 100 == 0:
            print(f"{k:3} Loss: {model.loss(distances).data:5.3f}")
    return model


class TestMDS1D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.distances = read_distances()
        cls.items = set(i for pair in cls.distances.keys() for i in pair)

    def d(self, model, a, b):
        p1 = point_values(model(a))
        p2 = point_values(model(b))
        return (sum((a-b)**2 for a, b in zip(p1, p2)))**0.5

    def model_distances(self, model):
        # Test case 1: Ljubljana-Trbovlje vs Ljubljana-Laško
        d1 = self.d(model, 'Ljubljana', 'Trbovlje')
        d2 = self.d(model, 'Ljubljana', 'Laško')
        self.assertLess(d1, d2, "Distance Ljubljana-Trbovlje should be less than Ljubljana-Laško")

        # Test case 2: Postojna-Logatec vs Postojna-Borovnica
        d1 = self.d(model, 'Postojna', 'Logatec')
        d2 = self.d(model, 'Postojna', 'Borovnica')
        self.assertLess(d1, d2, "Distance Postojna-Logatec should be less than Postojna-Borovnica")

        # Test case 3: Divača-Pivka vs Divača-Postojna
        d1 = self.d(model, 'Divača', 'Pivka')
        d2 = self.d(model, 'Divača', 'Postojna')
        self.assertLess(d1, d2, "Distance Divača-Pivka should be less than Divača-Postojna")

        # Test case 4: Zidani Most-Laško vs Zidani Most-Orehova vas
        d1 = self.d(model, 'Zidani Most', 'Laško')
        d2 = self.d(model, 'Zidani Most', 'Orehova vas')
        self.assertLess(d1, d2, "Distance Zidani Most-Laško should be less than Zidani Most-Orehova vas")

        # Test case 5: Koper-Divača vs Koper-Pivka
        d1 = self.d(model, 'Koper', 'Divača')
        d2 = self.d(model, 'Koper', 'Pivka')
        self.assertLess(d1, d2, "Distance Koper-Divača should be less than Koper-Pivka")

        # Test case 6: Orehova vas-Maribor vs Laško-Maribor
        d1 = self.d(model, 'Orehova vas', 'Maribor')
        d2 = self.d(model, 'Laško', 'Maribor')
        self.assertLess(d1, d2, "Distance Orehova vas-Maribor should be less than Laško-Maribor")

    def test_1d(self):
        fails = 0
        for i in range(10):
            model = MDS1D(self.items)
            model = train(model, self.distances)
            try:
                self.model_distances(model)
            except AssertionError:
                fails += 1
        # MDS v 1 dimenziji ima na teh podatkih težave
        self.assertGreater(fails, 4)

    def test_2d(self):
        fails = 0
        for i in range(10):
            model = MDS2D(self.items)
            model = train(model, self.distances)
            try:
                self.model_distances(model)
            except AssertionError:
                fails += 1
        # MDS v 2 dimenzijah ima na teh podatkih težave
        self.assertLessEqual(fails, 1)

    def test_2d_reg(self):
        fails = 0
        for i in range(10):
            model = MDS2DReg(self.items)
            model = train(model, self.distances)
            try:
                self.model_distances(model)
            except AssertionError:
                fails += 1
        # MDS v 2 dimenzijah ima na teh podatkih težave
        self.assertLessEqual(fails, 1)

    def test_2d_reg_small_dimension(self):
        model = MDS2DReg(self.items)
        model = train(model, self.distances)
        for i in self.items:
            x, y = model(i)
            self.assertLess(abs(y.data), 0.001)


if __name__ == '__main__':
    unittest.main()