# ETL
import numpy as np
import pandas as pd
from datetime import datetime

# visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# modelling
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

# custom
from pull_data import save_data, read_data

# save_data(15,'data/raw_data.csv')

bom = read_data('data/raw_data.csv')

# Preprocessing
ws_cols = ['9am wind speed (km/h)', '3pm wind speed (km/h)']
bom[ws_cols] = bom[ws_cols].replace(['Calm'], 0).astype(float)

# bom = pd.get_dummies(bom)

# split data into train & test sets
bom_trn, bom_tst = np.split(bom, [int(0.90*len(bom))])

sns.set()
# plt.plot(bom_trn['Evaporation (mm)'], color = 'black')
# plt.plot(bom_tst['Evaporation (mm)'], color = 'red')
plt.plot(bom_trn['Maximum temperature (°C)'], color='black')
plt.plot(bom_tst['Maximum temperature (°C)'], color='red')
# plt.show()

# Autoregressive Moving Average (ARMA)
# model = SARIMAX(bom_trn['Maximum temperature (°C)'], order=(1, 0, 1))

# Autoregressive Integrated Moving Average (ARIMA)
model = ARIMA(bom_trn['Maximum temperature (°C)'], order=(range(1,20), 1, range(1,20)))

# Seasonal Autoregressive Integrated Moving Average (SARIMA)
# model = ARIMA(bom_trn['Maximum temperature (°C)'], order=(10, 1, 10), seasonal_order=(2,2,2,12))


model = model.fit()

y_pred = model.get_forecast(len(bom_tst.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = model.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = bom_tst.index
y_pred_out = y_pred_df["Predictions"] 
plt.plot(y_pred_out, color='green', label = 'Predictions')
plt.legend()
plt.show()