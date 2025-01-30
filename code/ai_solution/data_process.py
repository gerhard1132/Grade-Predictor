""" 
Main execution of all main steps in creating a neural network
"""

import pandas as pd
import random

from data_prep.scrapeData import scrape_data_to_csv
from data_prep.prepareData import clean_data

# Step 1: Scrape data to csv
csv_path = scrape_data_to_csv()

# Step 2: Remove outliers
cleaned_csv_path = clean_data(csv_path)

# Step 3: Train/ test split + 1 random activation
data_list = pd.read_csv(cleaned_csv_path).values.tolist()
random.shuffle(data_list)
test_size = 0.2
split_index = int(len(data_list) * (1 - test_size)) 

train_data = data_list[:split_index]
test_data = data_list[split_index:]
random_activation = test_data[random.randint(0, len(test_data)-1)]

cols = ["Socioeconomic Score", "Study Hours", "Sleep Hours", "Attendance (%)", "Grades"]
train_df = pd.DataFrame(train_data, columns=cols)
test_df = pd.DataFrame(test_data, columns=cols)
activation_df = pd.DataFrame([random_activation], columns=cols)

train_df.to_csv('../data/train/train_data.csv', index=False)
test_df.to_csv('../data/validate/test_data.csv', index=False)
activation_df.to_csv('../data/activation/activation_data.csv', index=False)
