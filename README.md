# BoM ML Project
This repo contains code related to weather prediction. The main code-base can be found under `code_arima` with draft applications located under `draft_apps`.

# Setup Requirements
This project requires Python 3, install python requirements located in `requirements.txt` by running the helper script `setupEnv.sh` which sets up a python environment.

# How do I run/use the API?
- open CLI in the parent directory
- if you used `setupEnv.sh` above, activate the python environment using `source pythonenv/bin/activate`
- run `python code_arima/api.py`
- open a browser and navigate to `http://localhost:8000/docs`
- at the time of writting, several endpoints exist:
  - `/list_features/`: returns a JSON array of columns which can be used as the target of a model
  - `/train_arima/`: train an ARIMA model using the target feature, this endpoint must be run before `/forecast/` in order to train a model to use for predictions
  - `/forecast/`: predict future value(s) for the target feature, `predict_range` == true will return n predictions whereas `predict_range` == false will return the nth prediction e.g. forecast next weeks' temperature or the temperature 1 week from now respectively

# Backlog
## TODO
### Write some documentation
Worth putting together a section in this README talking about how to actually use it.

### Location Selection
Store a list of BoM sensors to choose from, allowing users to predict weather conditions in locations other than Melbourne


## Done
### Rewrite preprocessing
With around 365 days worth of data. If n is the number of days to use as input per row, then it should be possible to have 365 - n rows to model.

### Generate rows in reverse
Guarantee that latest data always falls in a full set (e.g. reverse the column index?)

### Updating Data
Update existing CSV based on missing data rather than doing a complete download each time can rsync do this?

### Separate Preprocessing & Modelling
Really should pull modelling out into a new python file to clean up a bit.

### Reduce date cardinality
Extract day, month & year of each date as a separate column, therefore creating less unique values and providing 3 new columns of (useful?) data.

### Create function to predict target given a date
Given that we're only really asking for the date (as other variables are pulled from BoM) a user should be able to specify a date and have an estimated value returned.

# URLs
- https://builtin.com/data-science/time-series-forecasting-python
- https://www.analyticsvidhya.com/blog/2021/06/predictive-modelling-rain-prediction-in-australia-with-python/ --> the dataset used here only predicts "tomorrow" i.e. requires data for the day before
