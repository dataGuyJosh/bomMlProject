import pandas as pd
import numpy as np
from pull_data import get_data
from sklearn import preprocessing
from datetime import datetime

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error

obs_df = get_data(12)

# replace "Calm" with 0 and convert to int
features_ws = ['9am wind speed (km/h)', '3pm wind speed (km/h)']
obs_df[features_ws] = obs_df[features_ws].replace(['Calm'], 0).astype(int)

# label encode categorical features (One Hot probably better here)
features_le = ['Time of maximum wind gust', 'Direction of maximum wind gust ',
               '9am wind direction', '3pm wind direction']
le = preprocessing.LabelEncoder()
for feature in features_le:
    le.fit(obs_df[feature])
    obs_df[feature] = le.transform(obs_df[feature])

# fill null values with 0
obs_df = obs_df.fillna(0)

# Column to predict
target = 'Maximum temperature (°C)'
# target = 'Evaporation (mm)'
# target = '9am relative humidity (%)'
# target = 'Rainfall (mm)'

# Days Per Row: How many days data should an individual row contain?
dpr = 4
# drop Date
obs_df.drop('Date', axis=1, inplace=True)
# save current columns for dropping later
old_cols = obs_df.columns
# trying to add columns directly results in performance warnings
# appending to a list then concatenating works better
new_cols = []
# append previous days data as new columns
for col in obs_df.columns:
    for i in range(dpr):
        # obs_df[col+str(i)] = obs_df[col].shift(dpr - i)
        new_cols.append(pd.DataFrame(
            data={col+str(i): obs_df[col].shift(dpr - i)}))

# add new columns & remove old ones by redefining dataframe
obs_df = pd.concat(new_cols, axis=1)
# redefine target
target = target+str(dpr - 1)
# clear first dpr rows due to incomplete data
obs_df.dropna(inplace=True)
# fix index (probably not necessary)
obs_df.reset_index(drop=True, inplace=True)

X = obs_df.drop([target], axis=1)
y = obs_df[target]

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

# # testing best degree for polynomial, lowest mean squared error usually best
# for i in range(1, 11):
#     poly_model = PolynomialFeatures(degree=i)
#     poly_X = poly_model.fit_transform(X)
#     poly_model.fit(poly_X, y)
#     regr_model = LinearRegression()
#     regr_model.fit(poly_X, y)

#     k_fold = KFold(n_splits=10, shuffle=True)
#     y_pred = regr_model.predict(poly_X)
#     print(i, mean_squared_error(y, y_pred, squared=False))

k_fold = KFold(n_splits=10, shuffle=True)


def cv_models(models, cv):
    for model in models:
        scores = cross_val_score(model, X, y, cv=cv)
        print(model,
              '\nIndividual Scores:', scores,
              '\nAverage Score:', scores.mean(),
              '\nNumber of scores used in Average:', len(scores), '\n'
              )


cv_models([dt_model, multi_reg_model, regr_model], k_fold)