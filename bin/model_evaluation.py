import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Define paths
input_dir = 'results/model_fit_evaluation'
output_dir = 'results/model_fit_evaluation'
os.makedirs(output_dir, exist_ok=True)

# Load the model and test data
model = joblib.load(os.path.join(input_dir, 'random_forest_model.pkl'))
X_test = joblib.load(os.path.join(input_dir, 'X_test.pkl'))
y_test = joblib.load(os.path.join(input_dir, 'y_test.pkl'))

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
report = classification_report(y_test, y_pred, output_dict=False)
matrix = confusion_matrix(y_test, y_pred)

# Save the evaluation results
with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
    f.write(report)

pd.DataFrame(matrix).to_csv(os.path.join(output_dir, 'confusion_matrix.csv'), index=False)
