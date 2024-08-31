import os
import pandas as pd

# Define the directory for exploratory data analysis (EDA) results
eda_dir = 'results/exploratory_data_analysis'

# Create the directory if it doesn't exist
os.makedirs(eda_dir, exist_ok=True)


data = pd.read_csv('data/train.csv')

# Perform data exploration
data_head = data.head()
data_description = data.describe()

# Save the results
data_head.to_csv(f'{eda_dir}/data_head.csv', index=False)
data_description.to_csv(f'{eda_dir}/data_description.csv')
