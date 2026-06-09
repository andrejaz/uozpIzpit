import pandas as pd


class LinReg:
    pass


class PoissonReg(LinReg):
    pass


def train(model, xs, ys, learning_rate=0.001, n_epochs=1000, batch_size=10):
    pass


def read_data():
    df = pd.read_csv("data.csv")
    ys = df.iloc[:, -1].values
    X = df.iloc[:, :-1].values
    return X, ys


if __name__ == "__main__":
    X, ys = read_data()

    # razdeli podatke na učne in testne
    X_train, ys_train = X[::2], ys[::2]
    X_test, ys_test = X[1::2], ys[1::2]

    linreg = LinReg(n_inputs=X.shape[1])
    poissonreg = PoissonReg(n_inputs=X.shape[1])

    for model in [linreg, poissonreg]:
        model = train(model, X_train, ys_train, n_epochs=1000,
                      batch_size=20, learning_rate=0.01)
        print(model)
        mae = sum([abs(y - model(x).data) for x, y in zip(X_test, ys_test)]) / len(X_test)
        print(f"MAE on test set: {mae:.3f}")
