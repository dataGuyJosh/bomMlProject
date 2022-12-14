import requests
import pandas as pd
from io import StringIO
from datetime import datetime


def get_data(bom_url, n_months):
    obs_df = pd.DataFrame()
    today = datetime.today().strftime('%Y/%m')
    month_strs = pd.date_range(end=today, periods=n_months,
                               freq='MS').strftime('%Y%m').tolist()
                               
    # concatenate n months of data into dataframe
    for i in month_strs:
        i_bom_url = bom_url.replace('~', i)
        response = requests.get(i_bom_url).text
        month_df = pd.read_csv(
            StringIO(response), skiprows=5).iloc[:, 1:]
        obs_df = pd.concat([obs_df, month_df], ignore_index=True)

    return obs_df


def save_data(n_months, data_path):
    get_data(n_months).to_csv(data_path, index=False)


def read_data(data_path):
    return pd.read_csv(data_path)


def update_data(data_path):
    print('dooo sooomethiiing')
