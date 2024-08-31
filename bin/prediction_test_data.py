import os
import joblib
import pandas as pd

# Define paths
model_fit_eval_dir = 'results/model_fit_evaluation'
output_dir = 'results/predict_test_data'
os.makedirs(output_dir, exist_ok=True)

# Load the model and test data
model = joblib.load(os.path.join(model_fit_eval_dir, 'random_forest_model.pkl'))
X_test = joblib.load(os.path.join(model_fit_eval_dir, 'X_test.pkl'))

# Make predictions
test_predictions = model.predict(X_test)

# Save the predictions
predictions_df = pd.DataFrame(test_predictions, columns=['Predictions'])
predictions_df.to_csv(os.path.join(output_dir, 'test_predictions.csv'), index=False)
