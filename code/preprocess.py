import pandas as pd
from pull_data import get_data
from sklearn import preprocessing

from sklearn import linear_model
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

# # shift target down 1 day and drop null values i.e. predict target based on yesterday's entries
# obs_df['Target: '+target] = obs_df[target].shift(1)
# obs_df = obs_df.dropna()

# group results by 7 days
target_column = obs_df[target]
obs_df.Date = obs_df.index/7
obs_df.Date = obs_df.Date.astype(int)
obs_df = obs_df.groupby('Date').agg(list)
obs_df['target'] = target_column

# X = obs_df[features]
# y = obs_df['target']

# dt = DecisionTreeRegressor()
# dt = dt.fit(X, y)

# multi_reg = linear_model.LinearRegression()
# multi_reg.fit(X, y)

# k_fold = KFold(n_splits=10, shuffle=True)


# def cv_models(models, cv):
#     for model in models:
#         scores = cross_val_score(model, X, y, cv=cv)
#         print(model,
#               '\nIndividual Scores:', scores,
#               '\nAverage Score:', scores.mean(),
#               '\nNumber of scores used in Average:', len(scores), '\n'
#               )


# cv_models([dt, multi_reg], k_fold)
