from pull_data import get_data
from sklearn import preprocessing

from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score, KFold

obs_df = get_data(12)

features_le = ['Date', 'Time of maximum wind gust', 'Direction of maximum wind gust ',
               '9am wind direction', '3pm wind direction']
features_ws = ['9am wind speed (km/h)', '3pm wind speed (km/h)']

# replace "Calm" with 0 and convert to int
obs_df[features_ws] = obs_df[features_ws].replace(['Calm'], 0).astype(int)

# label encode categorical features (One Hot probably better here)
le = preprocessing.LabelEncoder()
for feature in features_le:
    le.fit(obs_df[feature])
    obs_df[feature] = le.transform(obs_df[feature])

obs_df = obs_df.fillna(0)

features = ['Date', 'Minimum temperature (°C)', 'Maximum temperature (°C)',
            'Rainfall (mm)', 'Evaporation (mm)', 'Sunshine (hours)',
            'Direction of maximum wind gust ', 'Speed of maximum wind gust (km/h)',
            'Time of maximum wind gust', '9am Temperature (°C)',
            '9am relative humidity (%)', '9am cloud amount (oktas)',
            '9am wind direction', '9am wind speed (km/h)', '9am MSL pressure (hPa)',
            '3pm Temperature (°C)', '3pm relative humidity (%)',
            '3pm cloud amount (oktas)', '3pm wind direction',
            '3pm wind speed (km/h)', '3pm MSL pressure (hPa)']

target = 'Maximum temperature (°C)'

features.remove(target)

X = obs_df[features]
y = obs_df[target]

dt = DecisionTreeRegressor()
dt = dt.fit(X, y)

multi_reg = linear_model.LinearRegression()
multi_reg.fit(X, y)

k_fold = KFold(n_splits=5, shuffle=True)

for clf in [dt, multi_reg]:
    scores = cross_val_score(clf, X, y, cv=k_fold)
    print(clf,
          '\nIndividual Scores:', scores,
          '\nAverage Score:', scores.mean(),
          '\nNumber of scores used in Average:', len(scores), '\n'
          )
