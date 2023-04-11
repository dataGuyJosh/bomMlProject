import pandas as pd
from sklearn import preprocessing


def check_cardinality(df):
    categorical_features = [col for col in df.columns if df[col].dtype == 'O']
    unique_categories = {}
    for feature in categorical_features:
        unique_categories[feature] = len(df[feature].unique())
        print("Cardinality of {}: {}".format(
            feature, unique_categories[feature]))

    return unique_categories


def check_nulls(df):
    return df.isnull().sum()


def categorical_feature_handler(df):
    # replace "Calm" with 0 and convert to int
    features_ws = ['9am wind speed (km/h)', '3pm wind speed (km/h)']
    df[features_ws] = df[features_ws].replace(['Calm'], 0).astype(int)

    # label encode categorical features (One Hot probably better here)
    features_le = ['Direction of maximum wind gust ',
                   '9am wind direction', '3pm wind direction']
    le = preprocessing.LabelEncoder()
    for feature in features_le:
        le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df


def preprocess(target, dpr, obs_df):
    # target: feature to predict
    # dpr: days per row i.e. how many days data should an individual row contain?
    # obs_df: environmental observation dataframe

    # drop last row due to null data (temporary)
    obs_df.drop(obs_df.tail(1).index, inplace=True)

    categorical_feature_handler(obs_df)

    # propagate last valid observation forward to next valid
    # obs_df = obs_df.fillna(method='ffill')

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
