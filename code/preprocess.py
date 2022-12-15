import pandas as pd
from sklearn import preprocessing


def check_cardinality(df):
    categorical_features = [col for col in df.columns if df[col].dtype == 'O']
    for feature in categorical_features:
        n_categories = len(df[feature].unique())
        print("Cardinality of {}: {}".format(feature, n_categories))


def check_nulls(df):
    # categorical_features = [col for col in df.columns if df[col].dtype == 'O']
    print(df.isnull().sum())


def preprocess(target, dpr, obs_df):
    # target: feature to predict
    # dpr: days per row i.e. how many days data should an individual row contain?
    # obs_df: environmental observation dataframe

    # replace "Calm" with 0 and convert to int
    features_ws = ['9am wind speed (km/h)', '3pm wind speed (km/h)']
    obs_df[features_ws] = obs_df[features_ws].replace(['Calm'], 0).astype(int)

    # label encode categorical features (One Hot probably better here)
    features_le = ['Direction of maximum wind gust ',
                   '9am wind direction', '3pm wind direction']
    le = preprocessing.LabelEncoder()
    for feature in features_le:
        le.fit(obs_df[feature])
        obs_df[feature] = le.transform(obs_df[feature])

    # propagate last valid observation forward to next valid
    obs_df = obs_df.fillna(method='ffill')

    # store Date column for later use
    obs_Date = pd.to_datetime(obs_df.Date)
    # drop high cardinality columns
    obs_df.drop(['Date', 'Time of maximum wind gust'], axis=1, inplace=True)
    # trying to add columns directly results in performance warnings
    # appending to a list then concatenating works better
    new_cols = []
    # append previous dpr days worth of data as new columns
    for col in obs_df.columns:
        for i in range(dpr):
            new_cols.append(pd.DataFrame(
                data={col+str(i): obs_df[col].shift(dpr - i - 1)}))

    # add new columns & remove old ones by redefining dataframe
    obs_df = pd.concat(new_cols, axis=1)
    # reduce cardinality by splitting date into day/month/year
    obs_df = obs_df.assign(
        day=obs_Date.dt.day,
        month=obs_Date.dt.month,
        year=obs_Date.dt.year
    )
    # clear first dpr rows due to incomplete data
    obs_df.dropna(inplace=True)
    # fix index (probably not necessary)
    obs_df.reset_index(drop=True, inplace=True)
    obs_df.rename(columns={target+str(dpr - 1): 'target'}, inplace=True)
    return obs_df
