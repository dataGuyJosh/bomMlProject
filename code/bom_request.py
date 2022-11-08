import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
from sklearn import tree, preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, LeaveOneOut, LeavePOut, ShuffleSplit
bom_url = 'http://reg.bom.gov.au/fwo/IDV60801/IDV60801.95936.json'
df_response = pd.read_json(StringIO(requests.get(bom_url).text), orient='index')
obs_df = pd.DataFrame(df_response.data.observations)

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

clf = DecisionTreeRegressor()
clf = clf.fit(X, y)

# # tree.plot_tree(dt, feature_names=features)
# # plt.show()

k_fold = KFold(n_splits=10)
scores = cross_val_score(clf, X, y, cv=k_fold)
print('K-Fold',
      '\nIndividual Scores:', scores,
      '\nAverage Score:', scores.mean(),
      '\nNumber of scores used in Average:', len(scores))

# Shuffle Split
ss = ShuffleSplit(train_size=0.6, test_size=0.3, n_splits=10)
scores = cross_val_score(clf, X, y, cv=ss)

print('\nShuffle Split',
      '\nIndividual Scores:', scores,
      '\nAverage Score:', scores.mean(),
      '\nNumber of scores used in Average:', len(scores))