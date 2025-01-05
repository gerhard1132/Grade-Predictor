""" 
Performing vairous outlier removal techniques to imporve/ ensure better quality of our dataset
"""

import pandas as pd


path_to_data = "../data/"

def clean_data(csv_path):
    
    # Open data and split columns 
    data = pd.read_csv(csv_path)

    # Convert data to numeric
    all_columns = [
        "Socioeconomic Score", 
        "Study Hours", 
        "Sleep Hours", 
        "Attendance (%)", 
        "Grades"
    ]
    
    for column in all_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Drop rows with NaN values in x or y
    data = data.dropna(subset=all_columns)

    remove_atIndex = []

    for col_name, col_series in data.items():
        Q1 = col_series.quantile(0.25)
        Q3 = col_series.quantile(0.75)
        IQR = Q3 - Q1

        threshold = 2
        outliers = col_series[(col_series < Q1 - threshold * IQR) | (col_series > Q3 + threshold * IQR)]
        remove_atIndex.extend(outliers.index)
    data = data.drop(remove_atIndex)

    # Saving clean data
    cleaned_csv_path = path_to_data + 'joint_data_collection.csv'
    data.to_csv(cleaned_csv_path, index=False)
    
    return cleaned_csv_path
