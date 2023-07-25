# # import requests
# # from bs4 import BeautifulSoup

# # def get_weather_station_sensor_ids(url):
# #     response = requests.get(url)

# #     if response.status_code == 200:
# #         soup = BeautifulSoup(response.content, "html.parser")
# #         sensor_ids = []

# #         for link in soup.select("a[href*='IDCJDW']"):
# #             sensor_id = link["href"].split("/")[-1].split(".")[0]
# #             sensor_ids.append(sensor_id)

# #         return sensor_ids

# #     else:
# #         print("Failed to fetch data from the website.")
# #         return []

# # if __name__ == "__main__":
# #     sensor_ids = []
# #     for i in range(1,9):
# #         url = f'http://reg.bom.gov.au/climate/dwo/IDCJDW030{i}.shtml'
# #         sensor_ids.append(get_weather_station_sensor_ids(url))
# #     print("Available weather station sensor IDs:")
# #     print(sensor_ids)

# import requests
# from bs4 import BeautifulSoup
# import re

# def get_weather_stations(url):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.content, 'html.parser')

#     stations = []
#     table = soup.find('table', class_='links')
#     if table:
#         rows = table.find_all('tr')[1:]  # Skip the header row
#         for row in rows:
#             name = row.find('th').find('a').text.strip()
#             href = row.find('th').find('a')['href']
#             station_id = re.search(r'IDCJDW(\d+)\.', href).group(1)
#             stations.append({'Name': name, 'ID': f'IDCJDW{station_id}'})

#     return stations

# def find_letter_ranges(url):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.content, 'html.parser')
#     letter_ranges = []

#     content_div = soup.find('div', class_='content')
#     if content_div:
#         h1_element = content_div.find('h1')
#         if h1_element:
#             letter_range_text = h1_element.text.strip()
#             letter_range_match = re.search(r"\((\w+)\s+-\s+(\w+)\)", letter_range_text)
#             if letter_range_match:
#                 start_letter = letter_range_match.group(1)
#                 end_letter = letter_range_match.group(2)
#                 letter_ranges.append({'LetterRange': f"{start_letter} - {end_letter}", 'URL': url})

#     return letter_ranges

# def find_all_stations(base_urls):
#     for i in base_urls:
#         letter_ranges = find_letter_ranges(i)
#         all_stations = []
#         for letter_range_info in letter_ranges:
#             letter_range = letter_range_info['LetterRange']
#             url = letter_range_info['URL']
#             print(f"Extracting stations for range: {letter_range}")
#             stations = get_weather_stations(url)
#             all_stations.extend(stations)

#         for station in all_stations:
#             print(f"Name: {station['Name']}, ID: {station['ID']}")

# urlList = [
#     'http://reg.bom.gov.au/climate/dwo/IDCJDW0301.shtml'
# ]

# def generate_urls(base_url, min, max):
#     urls = []
#     for i in range(min, max):
#         url = base_url[:-7] + f"{i}.shtml"
#         urls.append(url)
#     return urls

# if __name__ == "__main__":
#     find_all_stations(generate_urls('http://reg.bom.gov.au/climate/dwo/IDCJDW0301.shtml',1,9))

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