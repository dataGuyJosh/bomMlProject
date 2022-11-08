import requests
import pandas as pd
from io import StringIO


def get_request(url):
    response = requests.get(url).text
    df_response = pd.read_json(StringIO(response), orient='index')
    return pd.DataFrame(df_response.data.observations)
