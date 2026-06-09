import unittest
import pandas as pd

from poisson import LinReg, PoissonReg, train


def read_data():
    df = pd.read_csv("data.csv")
    ys = df.iloc[:, -1].values
    X = df.iloc[:, :-1].values
    return X, ys


class TestPoisson(unittest.TestCase):

    def test_linreg(self):
        X, ys = read_data()
        X_train, ys_train = X[::2], ys[::2]
        X_test, ys_test = X[1::2], ys[1::2]
        model = LinReg(n_inputs=X.shape[1])
        model = train(model, X_train, ys_train, n_epochs=1000, batch_size=20, learning_rate=0.01)
        mae = sum([abs(y - model(x).data) for x, y in zip(X_test, ys_test)]) / len(X_test)
        self.assertTrue(20 < mae < 30)

    def test_poisson(self):
        X, ys = read_data()
        X_train, ys_train = X[::2], ys[::2]
        X_test, ys_test = X[1::2], ys[1::2]
        model = PoissonReg(n_inputs=X.shape[1])
        model = train(model, X_train, ys_train, n_epochs=1000, batch_size=20, learning_rate=0.01)
        mae = sum([abs(y - model(x).data) for x, y in zip(X_test, ys_test)]) / len(X_test)
        self.assertTrue(mae < 3.5)


if __name__ == '__main__':
    unittest.main()