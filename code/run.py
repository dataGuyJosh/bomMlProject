import json
import numpy as np
import pandas as pd
from pull_data import get_data
from preprocess import check_cardinality, preprocess
import matplotlib.pyplot as plt
from model import check_feature_importance, fit_models, cross_validate_models

# read configuration variables
config = json.load(open('config.json'))

# pull 12 months data
raw_obs_df = get_data(config['bom_url'], 12)

check_cardinality(raw_obs_df)

obs_df = preprocess(config['target'], raw_obs_df)
X = obs_df.drop(['target'], axis=1)
y = obs_df['target']

importance = pd.Series(check_feature_importance(X, y),index=X.columns)
importance.nlargest(10).plot(kind='barh')
plt.show()

models = fit_models(X, y)
scores = cross_validate_models(models, 10, X, y)

print(np.mean(scores, axis=1))
