# ETL
import numpy as np
import pandas as pd
from datetime import datetime

# visualisation
import matplotlib.pyplot as plt

# modelling
from statsmodels.tsa.arima.model import ARIMA

# custom
from pull_data import read_data, update_data

update_data(12, 'data/raw_data.csv')
bom = read_data('data/raw_data.csv')

target_col = 'Maximum temperature (°C)'
# target_col = 'Evaporation (mm)'

# Preprocessing
ws_cols = ['9am wind speed (km/h)', '3pm wind speed (km/h)']
bom[ws_cols] = bom[ws_cols].replace(['Calm'], 0).astype(float)

# bom = pd.get_dummies(bom)

# split data into train & test sets
trn, tst = np.split(bom, [int(0.90*len(bom))])

# Autoregressive Integrated Moving Average (ARIMA)
model = ARIMA(trn[target_col], order=(range(1, 40), 1, range(1, 40)))

# Seasonal Autoregressive Integrated Moving Average (SARIMA)
# model = ARIMA(trn[target_col], order=(10, 1, 10), seasonal_order=(2,2,2,12))

model = model.fit()

prd = model.get_forecast(len(tst.index)*3)
prd_df = prd.conf_int(alpha=0.05)
prd_df["Predictions"] = model.predict(
    start=prd_df.index[0], end=prd_df.index[-1])
# prd_df.index = tst.index
trn[target_col].plot(color='black', label='Train')
tst[target_col].plot(color='red', label='Test')
prd_df['Predictions'].plot(color='green', label='Predictions')
plt.legend()
plt.show()
