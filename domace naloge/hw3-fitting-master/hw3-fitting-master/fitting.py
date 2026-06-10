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
        #standardizacija / normalizacija 
        self.mean = np.mean(X,axis=0)
        self.std = np.std(X, axis=0)

        X_standardized = (X-self.mean)/self.std

        if self.regularization == "L2":
            #Ridge
            self.model = Ridge(alpha=self.alpha)
        else:
            #Lasso
            self.model = Lasso(alpha=self.alpha)

        self.model.fit(X_standardized, ys)

    def predict(self, X):
        X_standardized = (X-self.mean)/self.std
        return self.model.predict(X_standardized)

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
    #ocenit moramo s precnim preverjanjem in R^2
    #splitting v dva groupa 

    R = []

    for train_idx, test_idx in cv_splitter.split(X, ys):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = ys[train_idx], ys[test_idx]
        #fitting modela 
        model.fit(X_train, y_train)
        
        y_prediction = model.predict(X_test)
        y_test_mean = np.mean(y_test)

        R1 = sum((y_test-y_prediction)**2)
        R2 = sum((y_test-y_test_mean)**2)
        score = 1 - (R1/R2)

        R.append(score)

    return np.mean(R)

class FittedLinearStandardizedModel:

    def __init__(self, regularization="L2", alphas=[], cv_splitter=None):
        self.regularization = regularization
        self.alphas = alphas
        self.cv_splitter = cv_splitter

    def fit(self, X, ys):
        best_score = -np.inf
        self.best_alpha = 0
        for alpha in self.alphas:
            standard_model = LinearStandardizedModel(regularization=self.regularization, alpha=alpha)
            score = cross_val_r2(standard_model, X, ys, self.cv_splitter)

            if score > best_score:
                best_score = score
                self.best_alpha = alpha
            
        self.f_model = LinearStandardizedModel(regularization=self.regularization, alpha=self.best_alpha)
        self.f_model.fit(X, ys)
    def predict(self, X):
        return self.f_model.predict(X)


if __name__ == "__main__":
    X, ys, _ = load_data()

    mine, maxe = -3, 3
    fit_space = np.logspace(mine, maxe, num=(maxe-mine)*5 + 1)

    cv_splitter = CVSplitter(5, 0)
    # cv_splitter.split(X,ys)
    # ind = cv_splitter.get_n_splits(X,ys)
    # print(f"IND: {ind}")
    fitted_model = FittedLinearStandardizedModel(regularization="L1", alphas=fit_space, cv_splitter=cv_splitter)
    print("my", cross_val_r2(fitted_model, X, ys, cv_splitter=cv_splitter))

    fitted_model.fit(X, ys)
    print(fitted_model.best_alpha, fitted_model.best_score)
    print(fitted_model.predict(X[:5]))
