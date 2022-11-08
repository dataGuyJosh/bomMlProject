from sklearn import preprocessing
from pull_data import get_request

from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, LeaveOneOut, LeavePOut, ShuffleSplit
from sklearn.tree import DecisionTreeRegressor
bom_url = 'http://reg.bom.gov.au/fwo/IDV60801/IDV60801.95936.json'

obs_df = get_request(bom_url)

features = ['delta_t', 'gust_kmh', 'gust_kt', 'dewpt', 'press',
            'press_msl', 'press_qnh', 'rain_trace', 'rel_hum',
            'vis_km', 'wind_dir', 'wind_spd_kmh', 'wind_spd_kt']

# label encode wind direction
le = preprocessing.LabelEncoder()
le.fit(obs_df['wind_dir'])
obs_df['wind_dir'] = le.transform(obs_df['wind_dir'])

obs_df[features] = obs_df[features].astype(float)

X = obs_df[features]
y = obs_df['air_temp']

dt_clf = DecisionTreeRegressor()
dt_clf = dt_clf.fit(X, y)

for clf in [dt_clf]:
    k_fold = KFold(n_splits=10, shuffle=True)
    scores = cross_val_score(clf, X, y, cv=k_fold)
    print(clf,
          '\nIndividual Scores:', scores,
          '\nAverage Score:', scores.mean(),
          '\nNumber of scores used in Average:', len(scores))
