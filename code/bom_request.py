import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree, preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, LeaveOneOut, LeavePOut, ShuffleSplit
bom_url = 'http://reg.bom.gov.au/fwo/IDV60801/IDV60801.95936.json'
df_response = pd.read_json(requests.get(bom_url).text, orient='index')
obs_df = pd.DataFrame(df_response.data.observations)

features = ['delta_t', 'gust_kmh', 'gust_kt', 'dewpt', 'press',
            'press_msl', 'press_qnh', 'rain_trace', 'rel_hum',
            'vis_km', 'wind_dir', 'wind_spd_kmh', 'wind_spd_kt']

# label encode wind direction
le = preprocessing.LabelEncoder()
le.fit(obs_df['wind_dir'])
obs_df['wind_dir'] = le.transform(obs_df['wind_dir'])

obs_df[features] = obs_df[features].astype(float)

x = obs_df[features]
y = obs_df['air_temp']

dt = DecisionTreeRegressor()
dt = dt.fit(x, y)

# tree.plot_tree(dt, feature_names=features)
# plt.show()

x = x.values.tolist()
y = np.array(y.values.tolist())

loo = LeaveOneOut()
scores = cross_val_score(dt, x, y, cv=loo)
print('Individual Scores:', scores,
      '\nAverage Score:', scores.mean(),
      '\nNumber of scores used in Average:', len(scores))
