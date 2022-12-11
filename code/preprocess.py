import json
import pandas as pd
from sklearn import preprocessing
from datetime import datetime

def preprocess(target, obs_df):
    # replace "Calm" with 0 and convert to int
    features_ws = ['9am wind speed (km/h)', '3pm wind speed (km/h)']
    obs_df[features_ws] = obs_df[features_ws].replace(['Calm'], 0).astype(int)

    # label encode categorical features (One Hot probably better here)
    features_le = ['Time of maximum wind gust', 'Direction of maximum wind gust ',
                '9am wind direction', '3pm wind direction']
    le = preprocessing.LabelEncoder()
    for feature in features_le:
        le.fit(obs_df[feature])
        obs_df[feature] = le.transform(obs_df[feature])

    # fill null values with 0
    obs_df = obs_df.fillna(0)

    # How many days data should an individual row contain?
    dpr = json.load(open('config.json'))['days_per_row']
    # drop Date
    obs_df.drop('Date', axis=1, inplace=True)
    # save current columns for dropping later
    old_cols = obs_df.columns
    # trying to add columns directly results in performance warnings
    # appending to a list then concatenating works better
    new_cols = []
    # append previous days data as new columns
    for col in obs_df.columns:
        for i in range(dpr):
            # obs_df[col+str(i)] = obs_df[col].shift(dpr - i)
            new_cols.append(pd.DataFrame(
                data={col+str(i): obs_df[col].shift(dpr - i)}))

    # add new columns & remove old ones by redefining dataframe
    obs_df = pd.concat(new_cols, axis=1)
    # clear first dpr rows due to incomplete data
    obs_df.dropna(inplace=True)
    # fix index (probably not necessary)
    obs_df.reset_index(drop=True, inplace=True)

    obs_df.rename(columns={target+str(dpr - 1): 'target'}, inplace=True)

    return obs_df