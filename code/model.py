from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error


def fit_models(X, y):
    # Decision Tree
    dt_model = DecisionTreeRegressor()
    dt_model = dt_model.fit(X, y)

    # Multiple (Linear) Regression
    multi_reg_model = LinearRegression()
    multi_reg_model.fit(X, y)

    # Multivariate Polynomial Regression
    poly_model = PolynomialFeatures(degree=2)
    poly_X = poly_model.fit_transform(X)
    poly_model.fit(poly_X, y)
    regr_model = LinearRegression()
    regr_model.fit(poly_X, y)

    return [dt_model, multi_reg_model, regr_model]


def cross_validate_models(models, splits, X, y):
    k_fold = KFold(n_splits=splits, shuffle=True)
    model_scores = []
    for model in models:
        scores = cross_val_score(model, X, y, cv=k_fold)
        model_scores.append(scores.tolist())
    return model_scores


def test_polynomial(poly_deg, X, y):
    # testing best degree for polynomial, lowest mean squared error usually best
    poly_model = PolynomialFeatures(degree=poly_deg)
    poly_X = poly_model.fit_transform(X)
    poly_model.fit(poly_X, y)
    regr_model = LinearRegression()
    regr_model.fit(poly_X, y)

    k_fold = KFold(n_splits=10, shuffle=True)
    y_pred = regr_model.predict(poly_X)
    return mean_squared_error(y, y_pred, squared=False)
