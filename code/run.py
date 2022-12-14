import json
import numpy as np
from pull_data import get_data
from preprocess import check_cardinality, preprocess
from model import check_feature_importance, fit_generic_models, fit_mpr_model, cross_validate_models, predict_date

# read configuration variables
config = json.load(open('config.json'))

# pull 12 months data
raw_obs_df = get_data(config['bom_url'], 12)

# check_cardinality(raw_obs_df)

obs_df = preprocess(config['target'], raw_obs_df)
X = obs_df.drop(['target'], axis=1)
y = obs_df['target']

# check_feature_importance(X,y)

models = fit_generic_models(
    ['DecisionTreeRegressor', 'ExtraTreesRegressor', 'LinearRegression'], X, y)
models.append(fit_mpr_model(X, y))


scores = cross_validate_models(models, 10, X, y)

print(scores)
# print(np.mean(scores, axis=1))

predict_date('2022-12-12',X,'foo')
