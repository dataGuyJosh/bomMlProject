import requests
import pandas as pd
from io import StringIO
from datetime import datetime

bom_url = 'http://reg.bom.gov.au/climate/dwo/~/text/IDCJDW3049.~.csv'


def get_data(n_months):
    obs_df = pd.DataFrame()
    today = datetime.today().strftime('%Y/%m')
    month_strs = pd.date_range(end=today, periods=n_months,
                               freq='M').strftime('%Y%m').tolist()

    # concatenate n months of data into dataframe
    for i in month_strs:
        i_bom_url = bom_url.replace('~', i)
        response = requests.get(i_bom_url).text
        month_df = pd.read_csv(
            StringIO(response), skiprows=5).iloc[:, 1:]
        obs_df = pd.concat([obs_df, month_df], ignore_index=True)

    return obs_df


obs_df = get_data(12)

obs_df.to_csv('obs_df.csv', index=False)
