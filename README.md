# bomMlProject
Putting together a weather prediction model (ideally with FastAPI) using BoM Weather data.

# URLs
https://builtin.com/data-science/time-series-forecasting-python
https://sparkbyexamples.com/pandas/pandas-group-dataframe-rows-list-groupby/
https://enjoymachinelearning.com/blog/multivariate-polynomial-regression-python/
https://www.analyticsvidhya.com/blog/2021/06/predictive-modelling-rain-prediction-in-australia-with-python/

# TODOs
## Rewrite preprocessing - Done!
With around 365 days worth of data. If n is the number of days to use as input per row, then it should be possible to have 365 - n rows to model.

## Generate rows in reverse - Done!
Guarantee that latest data always falls in a full set (e.g. reverse the column index?)

## Updating Data
Update existing CSV based on missing data rather than doing a complete download each time can rsync do this?

## Separate Preprocessing & Modelling - Done!
Really should pull modelling out into a new python file to clean up a bit.

## Reduce date cardinality - Done!
Extract day, month & year of each date as a separate column, therefore creating less unique values and providing 3 new columns of (useful?) data.

## Create function to predict target given a date - Done!
Given that we're only really asking for the date (as other variables are pulled from BoM) a user should be able to specify a date and have an estimated value returned.

## Handling Null Values
- figure out why most recent dates are being culled
- today's data omits some variables which (technically) cannot be finalised until the end of the day e.g. max temperature, sunshine hours etc...