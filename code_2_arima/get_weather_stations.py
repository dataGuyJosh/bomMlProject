import re
import requests
from bs4 import BeautifulSoup
import pandas as pd


def get_weather_stations(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        # Find the relevant table containing the weather station names and IDs
        table = soup.find("table", class_="links")

        # Extract station names and IDs into lists
        station_names = []
        station_ids = []
        for row in table.find_all("tr"):
            station_name = row.th.a.get_text()
            station_id = re.search(r'IDCJDW\d+', row.th.a["href"]).group()
            station_names.append(station_name)
            station_ids.append(station_id)

        # Create a DataFrame from the lists
        data = {"station_name": station_names, "station_id": station_ids}
        weather_df = pd.DataFrame(data)

        # Print the DataFrame (optional)
        return weather_df

    except requests.exceptions.RequestException as e:
        print("Error fetching the website:", e)


def generate_urls(base_url, mn, mx):
    urls = []
    if not mx:
        return [base_url]

    for i in range(mn, mx):
        url = base_url[:-9] + f"{i}.shtml"
        urls.append(url)
    return urls


states = [
    {
        'region': 'act',
        'url': 'http://reg.bom.gov.au/climate/dwo/IDCJDW0100.shtml',
        'mn': 0,
        'mx': 0
    },
    {
        'region': 'nsw',
        'url': 'http://reg.bom.gov.au/climate/dwo/IDCJDW0200.shtml',
        'mn': 201,
        'mx': 213
    },
    {
        'region': 'vic',
        'url': 'http://reg.bom.gov.au/climate/dwo/IDCJDW0300.shtml',
        'mn': 301,
        'mx': 308
    },
    {
        'region': 'qld',
        'url': 'http://reg.bom.gov.au/climate/dwo/IDCJDW0400.shtml',
        'mn': 401,
        'mx': 412
    },
    {
        'region': 'sa',
        'url': 'http://reg.bom.gov.au/climate/dwo/IDCJDW0500.shtml',
        'mn': 501,
        'mx': 508
    },
    {
        'region': 'wa',
        'url': 'http://reg.bom.gov.au/climate/dwo/IDCJDW0600.shtml',
        'mn': 601,
        'mx': 612
    },
    {
        'region': 'tas',
        'url': 'http://reg.bom.gov.au/climate/dwo/IDCJDW0700.shtml',
        'mn': 701,
        'mx': 705
    },
    {
        'region': 'nt',
        'url': 'http://reg.bom.gov.au/climate/dwo/IDCJDW0800.shtml',
        'mn': 801,
        'mx': 806
    },
    {
        'region': 'antarctica',
        'url': 'http://reg.bom.gov.au/climate/dwo/IDCJDW0920.shtml',
        'mn': 0,
        'mx': 0
    },
    {
        'region': 'offshoreIslands',
        'url': 'http://reg.bom.gov.au/climate/dwo/IDCJDW0940.shtml',
        'mn': 0,
        'mx': 0
    },
]

act_url = 'http://reg.bom.gov.au/climate/dwo/IDCJDW0100.shtml'
vic_url = 'http://reg.bom.gov.au/climate/dwo/IDCJDW0301.shtml'

result = pd.DataFrame()

for st in states:
    print(f"Getting stations for {st['region']}...")
    for url in generate_urls(st['url'], st['mn'], st['mx']):
        region_data = get_weather_stations(url)
        region_data.insert(0,'region', st['region'])
        result = pd.concat([result, region_data], ignore_index=True)

print(result)

result.to_csv('data/weather_stations.csv', index=False)