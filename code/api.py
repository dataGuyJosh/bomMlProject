import json
from pull_data import get_data
from preprocess import check_nulls, check_cardinality, preprocess
from model import check_feature_importance, fit_generic_models, fit_mpr_model, cross_validate_models, predict_date

import uvicorn
from fastapi import FastAPI
app = FastAPI()

# read configuration variables
config = json.load(open('config.json'))

# pull data from bom API
raw_obs_df = get_data(config['bom_url'], config['data_month_range'])

obs_df = preprocess(
    config['target'], config['days_per_row'], raw_obs_df.copy())
X = obs_df.drop(['target'], axis=1)
y = obs_df['target']

models = fit_generic_models(
    ['DecisionTreeRegressor', 'ExtraTreesRegressor', 'LinearRegression'], X, y)
# models.append(fit_mpr_model(X, y))


scores = cross_validate_models(models, 10, X, y)


@app.get('/model_scores/')
def get_model_scores():
    results = []
    for model in scores:
        results.append(list(model.values())[0])

    return results


@app.get('/predict_date/{date}')
def predict(date):
    return predict_date(date, X, models[1])[0]


@app.get('/stats/')
def get_stats(stat):
    if stat == 'nulls':
        return check_nulls(raw_obs_df).to_dict()
    elif stat == 'cardinality':
        return check_cardinality(raw_obs_df)
    elif stat == 'feature_importance':
        return check_feature_importance(X, y)


if __name__ == "__main__":
    uvicorn.run('api:app', app_dir='code', reload=True)
