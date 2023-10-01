import pandas as pd

from pull_data import get_data

data_path = 'data/all_stations.csv'

## Data Extraction
stations = pd.read_csv('data/weather_stations.csv')
result = pd.DataFrame()

for stn_id in stations['station_id']:
    print(stn_id)
    station_data = get_data(stn_id, 12)
    station_data.insert(0, 'station_id', stn_id)
    result = pd.concat([result, station_data], ignore_index=True)

result.to_csv(data_path, index=False)

## Data Transformation
raw_data = pd.read_csv(data_path)
print(raw_data)