import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from pull_data import save_data, read_data

# save_data(12,'data/raw_data.csv')

bom = read_data('data/raw_data.csv')

# Preprocessing
# Convert Date column to data frame index
bom.index = pd.to_datetime(bom['Date'], format='%Y-%m-%d')
del bom['Date']

ws_cols = ['9am wind speed (km/h)', '3pm wind speed (km/h)']
bom[ws_cols] = bom[ws_cols].replace(['Calm'], 0).astype(float)

print(bom.info())

bom = pd.get_dummies(bom)

# split data into train & test sets
bom_trn, bom_tst = np.split(bom, [int(0.95*len(bom))])

sns.set()
# plt.plot(bom_trn['Evaporation (mm)'], color = 'black')
# plt.plot(bom_tst['Evaporation (mm)'], color = 'red')
plt.plot(bom_trn['Maximum temperature (°C)'], color = 'black')
plt.plot(bom_tst['Maximum temperature (°C)'], color = 'red')
plt.show()

