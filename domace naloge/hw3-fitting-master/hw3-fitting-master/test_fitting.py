import unittest

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fitting import cross_val_r2, LinearStandardizedModel, FittedLinearStandardizedModel


def load_data():
    df = pd.read_csv("body-fat-brozek.csv")
    feature_names = df.columns[1:].tolist()
    ys = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    return X, ys, feature_names


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


mine, maxe = -3, 3
alpha_fit = np.logspace(mine, maxe, num=(maxe - mine) * 5 + 1)


class TestFitting(unittest.TestCase):

    def test_01_model_standardization(self):
        X, ys, _ = load_data()
        m = LinearStandardizedModel(regularization="L2", alpha=1000)
        m.fit(X, ys)
        pred_1 = m.predict(X)
        m.fit(X*3.14+300, ys)
        pred_strange = m.predict(X*3.14+300)
        np.testing.assert_almost_equal(pred_1, pred_strange)

        X, ys, _ = load_data()
        m = LinearStandardizedModel(regularization="L1", alpha=1)
        m.fit(X, ys)
        pred_1 = m.predict(X)
        m.fit(X*3.14+300, ys)
        pred_strange = m.predict(X*3.14+300)
        np.testing.assert_almost_equal(pred_1, pred_strange)

    def test_02_cross_val_r2(self):
        X, ys, _ = load_data()

        class Predict20:
            def fit(self, X, ys):
                pass
            def predict(self, X):
                return np.full((X.shape[0],), 20)

        m = Predict20()
        score = cross_val_r2(m, X, ys, cv_splitter=CVSplitter(3))
        self.assertAlmostEqual(score, -0.018891018189156544)

        m = LinearStandardizedModel(regularization="L2", alpha=1)
        score = cross_val_r2(m, X, ys, cv_splitter=CVSplitter(3))
        self.assertAlmostEqual(score, 0.6938474195530153)

    def test_fitted_parameters(self):
        X, ys, _ = load_data()
        fitted_model = FittedLinearStandardizedModel(regularization="L1", alphas=alpha_fit,
                                                     cv_splitter=CVSplitter(5))
        fitted_model.fit(X, ys)

        best_alpha = fitted_model.best_alpha
        direct = LinearStandardizedModel(regularization="L1", alpha=best_alpha)
        direct.fit(X, ys)

        np.testing.assert_equal(fitted_model.predict(X[:5]), direct.predict(X[:5]))

    def test_evaluation_L1(self):
        X, ys, _ = load_data()

        # sklearn part
        pipe = make_pipeline(StandardScaler(), Lasso())
        cvm = GridSearchCV(estimator=pipe, param_grid={"lasso__alpha": alpha_fit},
                           scoring="r2", cv=CVSplitter(5))
        s_score = np.mean(cross_val_score(cvm, X, ys, scoring="r2", cv=CVSplitter(5)))

        # out part
        fitted = FittedLinearStandardizedModel(regularization="L1", alphas=alpha_fit,
                                               cv_splitter=CVSplitter(5))
        score = cross_val_r2(fitted, X, ys, cv_splitter=CVSplitter(5))

        self.assertAlmostEqual(s_score, score)

    def test_fitting_L1(self):
        X, ys, _ = load_data()

        # sklearn part
        pipe = make_pipeline(StandardScaler(), Lasso())
        cvm = GridSearchCV(estimator=pipe, param_grid={"lasso__alpha": alpha_fit},
                           scoring="r2", cv=CVSplitter(5))
        cvm.fit(X, ys)
        s_best_alpha = cvm.best_params_["lasso__alpha"]
        s_pred5 = cvm.predict(X[:5])

        # our part
        fitted = FittedLinearStandardizedModel(regularization="L1", alphas=alpha_fit,
                                               cv_splitter=CVSplitter(5))
        fitted.fit(X, ys)
        best_alpha = fitted.best_alpha
        pred5 = fitted.predict(X[:5])

        self.assertEqual(s_best_alpha, best_alpha)
        np.testing.assert_equal(pred5, s_pred5)

    def test_evaluation_L2(self):
        X, ys, _ = load_data()

        # sklearn part
        pipe = make_pipeline(StandardScaler(), Ridge())
        cvm = GridSearchCV(estimator=pipe, param_grid={"ridge__alpha": alpha_fit},
                           scoring="r2", cv=CVSplitter(5))
        s_score = np.mean(cross_val_score(cvm, X, ys, scoring="r2", cv=CVSplitter(5)))

        # our part
        fitted = FittedLinearStandardizedModel(regularization="L2", alphas=alpha_fit,
                                               cv_splitter=CVSplitter(5))
        score = cross_val_r2(fitted, X, ys, cv_splitter=CVSplitter(5))

        self.assertAlmostEqual(s_score, score)

    def test_fitting_L2(self):
        X, ys, _ = load_data()
        pipe = make_pipeline(StandardScaler(), Ridge())
        cvm = GridSearchCV(estimator=pipe, param_grid={"ridge__alpha": alpha_fit},
                           scoring="r2", cv=CVSplitter(5))
        cvm.fit(X, ys)
        s_best_alpha = cvm.best_params_["ridge__alpha"]
        s_pred5 = cvm.predict(X[:5])

        fitted = FittedLinearStandardizedModel(regularization="L2", alphas=alpha_fit,
                                               cv_splitter=CVSplitter(5))
        fitted.fit(X, ys)
        best_alpha = fitted.best_alpha
        pred5 = fitted.predict(X[:5])

        self.assertEqual(s_best_alpha, best_alpha)
        np.testing.assert_equal(pred5, s_pred5)


if __name__ == '__main__':
    unittest.main()