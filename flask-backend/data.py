# data.py

import pandas as pd

def load_data(data_path):
    # Load data from CSV and perform initial cleaning
    data = pd.read_csv(data_path)
    data['Debit'] = pd.to_numeric(data['Debit'], errors='coerce').fillna(0)
    data['Credit'] = pd.to_numeric(data['Credit'], errors='coerce').fillna(0)
    return data
# Test loading data
if __name__ == '__main__':
    data = load_data('data.csv')  # Make sure the path is correct
    print(data.head())  # Display the first few rows of the data
