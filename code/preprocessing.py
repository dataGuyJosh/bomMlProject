import pandas as pd
import pandasql as ps
from pull_data import get_request
from sklearn import preprocessing
from io import StringIO
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, LeaveOneOut, LeavePOut, ShuffleSplit
from sklearn.tree import DecisionTreeRegressor

bom_url_old = 'http://reg.bom.gov.au/fwo/IDV60801/IDV60801.95936.json'
bom_url = 'http://reg.bom.gov.au/climate/dwo/202210/text/IDCJDW3049.202210.csv'

obs_df = pd.read_csv(
    StringIO(get_request(bom_url)), skiprows=5).iloc[:, 1:]


features_le = ['Direction of maximum wind gust ', '9am wind direction', '3pm wind direction']
features_ws = ['9am wind speed (km/h)','3pm wind speed (km/h)']

# replace "Calm" with 0 and convert to int
obs_df[features_ws] = obs_df[features_ws].replace(['Calm'],0).astype(int)
# label encode categorical features (One Hot probably better here)
le = preprocessing.LabelEncoder()
for feature in features_le:
    le.fit(obs_df[feature])
    obs_df[feature] = le.transform(obs_df[feature])

# X = obs_df[features]
# y = obs_df['air_temp']

# dt_clf = DecisionTreeRegressor()
# dt_clf = dt_clf.fit(X, y)

# for clf in [dt_clf]:
#     k_fold = KFold(n_splits=10, shuffle=True)
#     scores = cross_val_score(clf, X, y, cv=k_fold)
#     print(clf,
#           '\nIndividual Scores:', scores,
#           '\nAverage Score:', scores.mean(),
#           '\nNumber of scores used in Average:', len(scores))
