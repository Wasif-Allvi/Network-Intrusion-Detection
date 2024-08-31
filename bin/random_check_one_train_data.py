import os
import joblib
import pandas as pd
import numpy as np

# Define paths
model_fit_eval_dir = 'results/model_fit_evaluation'
output_dir = 'results/predict_test_data'
os.makedirs(output_dir, exist_ok=True)

# Load the model
model = joblib.load(os.path.join(model_fit_eval_dir, 'random_forest_model.pkl'))

# Load the test data (or training data, depending on the check)
X_test = joblib.load(os.path.join(model_fit_eval_dir, 'X_test.pkl'))

# Randomly select a single row from the test set
random_row = X_test.sample(n=1, random_state=42)

# Make prediction on the random row
random_pred = model.predict(random_row)

# Convert the prediction to a scalar if it's an array with one element
random_pred = random_pred[0]

# Combine the features and the prediction into a DataFrame
random_check_df = pd.DataFrame({
    'Feature': random_row.columns,
    'Value': random_row.values.flatten(),
    'Prediction': [random_pred] * random_row.shape[1]
})

# Save the prediction result
random_check_df.to_csv(os.path.join(output_dir, 'random_check.csv'), index=False)
