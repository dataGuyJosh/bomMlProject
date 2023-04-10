import json
from pull_data import get_data
from preprocess import check_nulls, check_cardinality, preprocess
from model import check_feature_importance, fit_generic_models, fit_mpr_model, cross_validate_models, predict_date

# read configuration variables
config = json.load(open('config.json'))

# pull data from bom API
raw_obs_df = get_data(config['bom_url'], config['data_month_range'])

obs_df = preprocess(
    config['target'], config['days_per_row'], raw_obs_df.copy())
X = obs_df.drop(['target'], axis=1)
y = obs_df['target']

# check_nulls(raw_obs_df)
check_cardinality(raw_obs_df)
# check_feature_importance(X,y)

models = fit_generic_models(
    ['DecisionTreeRegressor', 'ExtraTreesRegressor', 'LinearRegression'], X, y)
# models.append(fit_mpr_model(X, y))


scores = cross_validate_models(models, 10, X, y)
# print(raw_obs_df, obs_df)
print(
    scores,
    predict_date('2023-01-02', X, models[1])
)
