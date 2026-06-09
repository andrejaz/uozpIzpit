import pandas as pd
import numpy as np

# Ridge and Lasso and the only things that you can import from scikit-learn
from sklearn.linear_model import Ridge, Lasso


def load_data():
    df = pd.read_csv("body-fat-brozek.csv")
    feature_names = df.columns[1:].tolist()
    ys = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    return X, ys, feature_names


class LinearStandardizedModel:

    def __init__(self, regularization="L2", alpha=1.0):
        self.regularization = regularization
        self.alpha = alpha

    def fit(self, X, ys):
        pass  # to implement

    def predict(self, X):
        pass  # to implement


class CVSplitter:
    """
    A simple Cross-validation splitter that is compatible
    with scikit-learn. Do not change it.
    """

    def __init__(self, folds, rseed=0):
        self.folds = folds
        self.rseed = rseed

    def split(self, X, ys, groups=None):
        rand = np.random.RandomState(self.rseed)
        l = np.arange(len(X)) % self.folds
        rand.shuffle(l)
        for i in range(self.folds):
            yield np.where(l != i)[0], np.where(l == i)[0]

    def get_n_splits(self, X, ys, groups=None):
        return self.folds


def cross_val_r2(model, X, ys, cv_splitter):
    pass  # to implement


class FittedLinearStandardizedModel:

    def __init__(self, regularization="L2", alphas=[], cv_splitter=None):
        self.regularization = regularization
        self.alphas = alphas
        self.cv_splitter = cv_splitter

    def fit(self, X, ys):
        pass  # to implement

    def predict(self, X):
        pass  # to implement


if __name__ == "__main__":
    X, ys, _ = load_data()

    mine, maxe = -3, 3
    fit_space = np.logspace(mine, maxe, num=(maxe-mine)*5 + 1)

    cv_splitter = CVSplitter(5, 0)
    fitted_model = FittedLinearStandardizedModel(regularization="L1", alphas=fit_space, cv_splitter=cv_splitter)
    print("my", cross_val_r2(fitted_model, X, ys, cv_splitter=cv_splitter))

    fitted_model.fit(X, ys)
    print(fitted_model.best_alpha, fitted_model.best_score)
    print(fitted_model.predict(X[:5]))
