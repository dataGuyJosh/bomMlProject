import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error


def check_feature_importance(X, y):
    etr_model = ExtraTreesRegressor()
    etr_model.fit(X, y)
    importance = pd.Series(etr_model.feature_importances_, index=X.columns)
    # importance.nlargest(10).plot(kind='barh')
    # plt.show()
    return importance


def fit_generic_models(models, X, y):
    fit_models = []
    for model in models:
        model = eval(model+'()').fit(X, y)
        fit_models.append(model)
    return fit_models


def fit_mpr_model(X, y):
    # Multivariate Polynomial Regression
    poly_model = PolynomialFeatures(degree=2)
    poly_X = poly_model.fit_transform(X)
    poly_model.fit(poly_X, y)
    regr_model = LinearRegression()
    regr_model.fit(poly_X, y)

    return regr_model


def predict_date(date, df, model):
    # date expected in format YYYY-MM-DD
    date = [int(i) for i in date.split('-')]
    input_variables = df[
        (df.year == date[0]) &
        (df.month == date[1]) &
        (df.day == date[2])
    ]
    return model.predict(input_variables)


def cross_validate_models(models, splits, X, y):
    k_fold = KFold(n_splits=splits, shuffle=False)
    model_scores = []
    for model in models:
        scores = cross_val_score(model, X, y, cv=k_fold)
        model_scores.append({model: scores.mean()})
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
