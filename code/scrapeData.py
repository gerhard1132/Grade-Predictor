""" 
Scraping data of students performance from https://www.kaggle.com/datasets/stealthtechnologies/predict-student-performance-dataset/data
"""

from bs4 import BeautifulSoup
import requests
import csv


path_to_data = "../data/"

def scrape_data_to_csv():

    # Getting site html via requests (now obsolete)
    site = requests.get('https://www.kaggle.com/datasets/stealthtechnologies/predict-student-performance-dataset/data')

    # Note: Altered this part due to dynamic loading of the table data
    # Our workaround: Loaded every data entry in a browser and then saved the html to the website in raw_site.html to emulate scraping

    # Extracting every data value and saving it in a csv
    all_data = []

    with open(path_to_data + "raw_site.html", "r") as site:
        html = site.read()
        soup = BeautifulSoup(html, features="html.parser")
        containing_data = soup.find("div", {"role": "rowgroup"})
        data_entries = containing_data.find_all("span")
        for entry in data_entries:
            value_elements = entry.find_all('td')
            values = map(lambda x: x.text, value_elements)
            all_data.append(list(values))

        csv_path = path_to_data + "allData.csv"
        with open(csv_path, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerows(all_data)
            
        return csv_path