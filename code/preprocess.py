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

features = obs_df.columns.tolist()

target = 'Maximum temperature (°C)'
# target = 'Rainfall (mm)'

features.remove('Date')
features.remove(target)

# shift target down 1 day and drop null values i.e. predict target based on yesterday's entries
# obs_df['target'] = obs_df[target].shift(1)
# obs_df = obs_df.dropna()

# create week index
obs_df['week_index'] = (obs_df.index/7).astype(int)
# store last value as weekly target
target_column = obs_df.groupby('week_index')[target].last()
# create day index
obs_df['day_index'] = pd.to_datetime(
    obs_df['Date'], format='%Y-%m-%d').dt.weekday

# TODO guarantee that latest data always falls in a full week (e.g. reverse the day index)
obs_df.drop(obs_df[obs_df['day_index'] == 6].index, inplace=True)

# pivot table such that rows are weeks and columns are telemetry per day
obs_df = pd.pivot_table(obs_df, index=['week_index'], columns=[
                        'day_index'], aggfunc='last')
obs_df['target'] = target_column

# drop incomplete weeks (consider alternative solutions such as fillna)
obs_df = obs_df.dropna()

# flatten columns
obs_df.columns = [''.join([str(c) for c in c_list])
                  for c_list in obs_df.columns.values]
obs_df = obs_df.reset_index().drop(['week_index'], axis=1)

X = obs_df.drop(['target'], axis=1)
y = obs_df.target

# # Decision Tree
# dt_model = DecisionTreeRegressor()
# dt_model = dt_model.fit(X, y)
# # Multiple (Linear) Regression
# multi_reg = LinearRegression()
# multi_reg.fit(X, y)
# # Multivariate Polynomial Regression
# poly_model = PolynomialFeatures(degree=5)
# poly_X = poly_model.fit_transform(X)
# poly_model.fit(poly_X, y)
# regr_model = LinearRegression()
# regr_model.fit(poly_X, y)

# # # testing best degree for polynomial, results suggest 5
# # # for i in range(1, 11):
# # #     poly_model = PolynomialFeatures(degree=i)
# # #     poly_X = poly_model.fit_transform(X)
# # #     poly_model.fit(poly_X, y)
# # #     regr_model = LinearRegression()
# # #     regr_model.fit(poly_X, y)

# # #     k_fold = KFold(n_splits=10, shuffle=True)
# # #     y_pred = regr_model.predict(poly_X)
# # #     print(i, mean_squared_error(y, y_pred, squared=False))

# k_fold = KFold(n_splits=10, shuffle=True)

# def cv_models(models, cv):
#     for model in models:
#         scores = cross_val_score(model, X, y, cv=cv)
#         print(model,
#               '\nIndividual Scores:', scores,
#               '\nAverage Score:', scores.mean(),
#               '\nNumber of scores used in Average:', len(scores), '\n'
#               )


# cv_models([dt_model, multi_reg, regr_model], k_fold)
