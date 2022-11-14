from sklearn.metrics import mean_squared_error
import pandas as pd
from pull_data import get_data
from sklearn import preprocessing

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score, KFold

obs_df = get_data(12)

# replace "Calm" with 0 and convert to int
features_ws = ['9am wind speed (km/h)', '3pm wind speed (km/h)']
obs_df[features_ws] = obs_df[features_ws].replace(['Calm'], 0).astype(int)

# label encode categorical features (One Hot probably better here)
features_le = ['Date', 'Time of maximum wind gust', 'Direction of maximum wind gust ',
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
obs_df['target'] = obs_df[target].shift(1)
obs_df = obs_df.dropna()

# # group results by 7 days
# target_column = obs_df[target]
# obs_df.Date = obs_df.index/7
# obs_df.Date = obs_df.Date.astype(int)
# obs_df = obs_df.groupby('Date').agg(list)
# obs_df['target'] = target_column

X = obs_df[features]
y = obs_df['target']

# Decision Tree
dt_model = DecisionTreeRegressor()
dt_model = dt_model.fit(X, y)
# Multiple Regression
multi_reg = LinearRegression()
multi_reg.fit(X, y)
# Multivariate Polynomial Regression
poly_model = PolynomialFeatures(degree=5)
poly_X = poly_model.fit_transform(X)
poly_model.fit(poly_X, y)
regr_model = LinearRegression()
regr_model.fit(poly_X, y)

for i in range(1, 11):
    poly_model = PolynomialFeatures(degree=i)
    poly_X = poly_model.fit_transform(X)
    poly_model.fit(poly_X, y)
    regr_model = LinearRegression()
    regr_model.fit(poly_X, y)
    
    k_fold = KFold(n_splits=10, shuffle=True)
    # scores = cross_val_score(regr_model, poly_X, y, cv=k_fold)
    y_pred = regr_model.predict(poly_X)
    # print(i, scores.mean(), mean_squared_error(y, y_pred, squared=False))
    print(i, mean_squared_error(y, y_pred, squared=False))

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
