# BoM ML Project
Putting together a weather prediction model (ideally with FastAPI) using BoM Weather data.

# URLs
- https://builtin.com/data-science/time-series-forecasting-python
- https://www.analyticsvidhya.com/blog/2021/06/predictive-modelling-rain-prediction-in-australia-with-python/ --> the dataset used here only predicts "tomorrow" i.e. requires data for the day before


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