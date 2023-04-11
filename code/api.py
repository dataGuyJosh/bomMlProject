import json
from pull_data import get_data
from preprocess import check_nulls, check_cardinality, preprocess
from model import check_feature_importance, fit_generic_models, fit_mpr_model, cross_validate_models, predict_date

import uvicorn
from fastapi import FastAPI, HTTPException
from starlette import status
app = FastAPI()

# read configuration variables
config = json.load(open('config.json'))

# pull data from bom API
raw_obs_df = get_data(config['bom_url'], config['data_month_range'])


def check_models_trained():
    if not 'obs_df' in globals():
        raise HTTPException(
            status_code=400, detail='Please train models first')


# Maximum temperature (°C)
# Rainfall (mm)
@app.post('/train_models/', status_code=status.HTTP_204_NO_CONTENT)
def train_models(target):
    global obs_df, X, y, models, scores
    obs_df = preprocess(
        target, config['days_per_row'], raw_obs_df.copy())
    X = obs_df.drop(['target'], axis=1)
    y = obs_df['target']

    models = fit_generic_models(
        ['DecisionTreeRegressor', 'ExtraTreesRegressor', 'LinearRegression'], X, y)
    # models.append(fit_mpr_model(X, y))

    scores = cross_validate_models(models, 10, X, y)


@app.get('/predict_date/')
def predict(date):
    check_models_trained()
    return predict_date(date, X, models[1])[0]


@app.get('/stats/')
def get_stats(stat):
    check_models_trained()
    if stat == 'nulls':
        return check_nulls(raw_obs_df).to_dict()
    elif stat == 'cardinality':
        return check_cardinality(raw_obs_df)
    elif stat == 'importance':
        return check_feature_importance(X, y)


@app.get('/model_scores/')
def get_model_scores():
    check_models_trained()
    results = []
    for model in scores:
        results.append(list(model.values())[0])

    return results


if __name__ == "__main__":
    uvicorn.run('api:app', app_dir='code', reload=True)
