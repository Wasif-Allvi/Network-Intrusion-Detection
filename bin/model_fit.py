import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.preprocessing import LabelEncoder

# Ensure the output directory exists
output_dir = 'results/model_fit_evaluation'
os.makedirs(output_dir, exist_ok=True)

# Load data
train_data = pd.read_csv('data/train.csv')

# Prepare the data
label_encoders = {}
for column in ['protocol_type', 'service', 'flag', 'class']:
    label_encoders[column] = LabelEncoder()
    train_data[column] = label_encoders[column].fit_transform(train_data[column])

X = train_data.drop('class', axis=1)
y = train_data['class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model and data splits
joblib.dump(model, os.path.join(output_dir, 'random_forest_model.pkl'))
joblib.dump(X_test, os.path.join(output_dir, 'X_test.pkl'))
joblib.dump(y_test, os.path.join(output_dir, 'y_test.pkl'))
