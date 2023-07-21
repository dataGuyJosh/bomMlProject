# import requests
# from bs4 import BeautifulSoup

# def get_weather_station_sensor_ids(url):
#     response = requests.get(url)

#     if response.status_code == 200:
#         soup = BeautifulSoup(response.content, "html.parser")
#         sensor_ids = []

#         for link in soup.select("a[href*='IDCJDW']"):
#             sensor_id = link["href"].split("/")[-1].split(".")[0]
#             sensor_ids.append(sensor_id)

#         return sensor_ids

#     else:
#         print("Failed to fetch data from the website.")
#         return []

# if __name__ == "__main__":
#     sensor_ids = []
#     for i in range(1,9):
#         url = f'http://reg.bom.gov.au/climate/dwo/IDCJDW030{i}.shtml'
#         sensor_ids.append(get_weather_station_sensor_ids(url))
#     print("Available weather station sensor IDs:")
#     print(sensor_ids)

import requests
from bs4 import BeautifulSoup
import re

def get_weather_stations(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    stations = []
    table = soup.find('table', class_='links')
    if table:
        rows = table.find_all('tr')[1:]  # Skip the header row
        for row in rows:
            name = row.find('th').find('a').text.strip()
            href = row.find('th').find('a')['href']
            station_id = re.search(r'IDCJDW(\d+)\.', href).group(1)
            stations.append({'Name': name, 'ID': f'IDCJDW{station_id}'})
    
    return stations

def find_letter_ranges(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    letter_ranges = []

    content_div = soup.find('div', class_='content')
    if content_div:
        h1_element = content_div.find('h1')
        if h1_element:
            letter_range_text = h1_element.text.strip()
            letter_range_match = re.search(r"\((\w+)\s+-\s+(\w+)\)", letter_range_text)
            if letter_range_match:
                start_letter = letter_range_match.group(1)
                end_letter = letter_range_match.group(2)
                letter_ranges.append({'LetterRange': f"{start_letter} - {end_letter}", 'URL': url})
    
    return letter_ranges

def main():
    base_url = "http://reg.bom.gov.au/climate/dwo/IDCJDW0301.shtml"
    letter_ranges = find_letter_ranges(base_url)
    all_stations = []
    for letter_range_info in letter_ranges:
        letter_range = letter_range_info['LetterRange']
        url = letter_range_info['URL']
        print(f"Extracting stations for range: {letter_range}")
        stations = get_weather_stations(url)
        all_stations.extend(stations)

    for station in all_stations:
        print(f"Name: {station['Name']}, ID: {station['ID']}")

if __name__ == "__main__":
    main()
