import requests
import pandas as pd
from io import StringIO
from datetime import datetime


def get_data(station_id='IDCJDW3049', n_months=12, end_month_offset=None):
    bom_url = f'http://reg.bom.gov.au/climate/dwo/~/text/{station_id}.~.csv'
    df = pd.DataFrame()
    today = datetime.today().strftime('%Y/%m')
    month_strs = pd.date_range(end=today, periods=n_months,
                               freq='MS').strftime('%Y%m').tolist()

    month_strs = month_strs[:end_month_offset]
    # concatenate n months of data into dataframe
    for i in month_strs:
        i_bom_url = bom_url.replace('~', i)
        response = requests.get(i_bom_url).text
        # try to read response as CSV, skip on failure
        try:
            month_df = pd.read_csv(
                StringIO(response), skiprows=5).iloc[:, 1:]
        except:
            print(f'skipped {station_id} {i}')
            continue

        df = pd.concat([df, month_df])

    return df


def read_data(data_path):
    return pd.read_csv(data_path, index_col='Date', parse_dates=True).asfreq('D')


def update_data(read_path, write_path=None):
    if not write_path:
        write_path = read_path

    old = pd.read_csv(read_path)
    new = get_data()

    # convert Date column to datetime for sorting
    old['Date'] = old['Date'].astype('datetime64[ns]')
    new['Date'] = new['Date'].astype('datetime64[ns]')

    # concat dataframes i.e. old above new
    # drop duplicate rows based on Date (new rows overwrite old rows)
    # write to CSV using Date as the index
    pd.concat([old, new]) \
        .drop_duplicates(subset='Date', keep='last') \
        .to_csv(write_path, index=False)