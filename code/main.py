""" 
Main execution of all main steps in creating a NN
"""

from scrapeData import scrape_data_to_csv
from prepareData import clean_data

# Step 1: Scrape data to csv
csv_path = scrape_data_to_csv()

# Step 2: Remove outliers
cleaned_csv_path = clean_data(csv_path)

# Step 3: ...