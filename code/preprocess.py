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

target = 'Maximum temperature (°C)'
# target = 'Rainfall (mm)'

# --- Something Spicey ---
# How many days data should an individual row contain?
days_per_row = 2
# create row index
obs_df['row_index'] = (obs_df.index/days_per_row).astype(int)
# store last value per row_index as target
target_column = obs_df.groupby('row_index')[target].last()
# create column index
obs_df['col_index'] = obs_df.index % days_per_row
# drop date
obs_df.drop('Date', axis=1, inplace=True)

# drop last day per row_index (as this is what we're trying to predict)
obs_df.drop(obs_df[obs_df['col_index'] == days_per_row - 1].index, inplace=True)

# pivot table such that rows are groups of days and columns are telemetry per day
obs_df = pd.pivot_table(obs_df, index=['row_index'], columns=[
                        'col_index'], aggfunc='last')
obs_df['target'] = target_column

# drop incomplete rows (consider alternative solutions such as fillna)
obs_df = obs_df.dropna()

# flatten columns
obs_df.columns = [''.join([str(c) for c in c_list])
                  for c_list in obs_df.columns.values]
obs_df = obs_df.reset_index().drop(['row_index'], axis=1)

X = obs_df.drop(['target'], axis=1)
y = obs_df.target

# Decision Tree
dt_model = DecisionTreeRegressor()
dt_model = dt_model.fit(X, y)
# Multiple (Linear) Regression
multi_reg_model = LinearRegression()
multi_reg_model.fit(X, y)
# Multivariate Polynomial Regression
poly_model = PolynomialFeatures(degree=5)
poly_X = poly_model.fit_transform(X)
poly_model.fit(poly_X, y)
regr_model = LinearRegression()
regr_model.fit(poly_X, y)

# # testing best degree for polynomial, results suggest 5
# for i in range(1, 11):
#     poly_model = PolynomialFeatures(degree=i)
#     poly_X = poly_model.fit_transform(X)
#     poly_model.fit(poly_X, y)
#     regr_model = LinearRegression()
#     regr_model.fit(poly_X, y)

#     k_fold = KFold(n_splits=10, shuffle=True)
#     y_pred = regr_model.predict(poly_X)
#     print(i, mean_squared_error(y, y_pred, squared=False))

k_fold = KFold(n_splits=5, shuffle=True)


def cv_models(models, cv):
    for model in models:
        scores = cross_val_score(model, X, y, cv=cv)
        print(model,
              '\nIndividual Scores:', scores,
              '\nAverage Score:', scores.mean(),
              '\nNumber of scores used in Average:', len(scores), '\n'
              )


cv_models([dt_model, multi_reg_model, regr_model], k_fold)