import json
from pull_data import get_data
from preprocess import preprocess
from model import fit_models, cross_validate_models

# read configuration variables
config = json.load(open('config.json'))

# pull 12 months data
raw_obs_df = get_data(config['bom_url'], 12)

obs_df = preprocess(config['target'], raw_obs_df)
X = obs_df.drop(['target'], axis=1)
y = obs_df['target']

models = fit_models(X, y)

cross_validate_models(models, X, y)