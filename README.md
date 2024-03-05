
# Network Intrusion Detection using Neural Networks


### Project Overview:

Welcome to the Intrusion Detection System (IDS) repository! This project implements a Neural Network-based IDS to classify network traffic into normal and anomalous activities. The model is trained on a labeled dataset, providing an effective solution for detecting potential security threats.

### Training Dataset (`Train_data.csv`)

- The training dataset consists of labeled instances, with features describing various aspects of network traffic.
- Features include `protocol_type`, `service`, `flag`, and others, which are transformed and utilized during training.
- The target variable, `class`, indicates whether an instance is classified as normal or represents an intrusion (anomaly).

### Test Dataset (`Test_data.csv`)

- The test dataset (`Test_data.csv`) is used for making predictions on unseen data.
- Ensure that the test data follows the same structure as the training data, excluding the target variable (`class`).
- The preprocessing steps, including label encoding and scaling, should be applied to the test data before making predictions.

## Dependencies

- Python 3.x
- The project relies on the following libraries: numpy, pandas, scikit-learn, tensorflow, keras
## Data Splitting Techniques

When working with machine learning models, it's crucial to split the available data into training and testing sets. This practice helps in evaluating the model's performance on unseen data and mitigates the risk of overfitting. In this project, we employ the `train_test_split` function from the `scikit-learn` library.

### Train-Test Split

The dataset undergoes a division into two main segments: a training set and a test set. The training set serves the purpose of training the model, while the test set remains reserved for assessing its performance. A common split ratio is 70-30 or 80-20, where the majority of the data contributes to training.

In our project, we opt for a 70-30 train-test split, allocating 70% of the data to the training set and reserving 30% for the test set.

### Split Dimensions

Following the train-test split, the dimensions of the resulting sets are as follows:

- **Training Set (X_train, y_train):**
  - X_train: (17634, 41)
  - y_train: (17634,)

- **Test Set (X_test, y_test):**
  - X_test: (7558, 41)
  - y_test: (7558,)

These dimensions offer insights into the size of each set, ensuring a substantial amount of data for effective training and robust model evaluation. The choice of split ratio can be adjusted based on specific project requirements and dataset characteristics.

## Neural Network Model Architecture

The Intrusion Detection System (IDS) employs a simple neural network architecture consisting of:

### Input Layer
- Neurons equal to the number of features in the dataset.

### Hidden Layers
- One or more hidden layers for capturing complex relationships.
- Commonly uses the 'relu' (Rectified Linear Unit) activation function.

### Output Layer
- One neuron for binary classification (e.g., normal or anomaly).
- 'Sigmoid' activation function for binary output.

### Model Compilation
- Binary crossentropy loss function.
- Adam optimizer with a learning rate of 0.001.
- Accuracy metric for evaluation.

### Training
- The model is trained on the labeled training dataset.

This architecture is adaptable to different datasets, allowing flexibility for customization based on specific project requirements.




## Model Training and Evaluation

### Training Process

The neural network model is trained using the labeled training dataset (`Train_data.csv`). The training process involves the following steps:

- **Data Preprocessing:**
    - There are no missing values in the dataset, ensuring a complete and reliable training process.
   - Categorical features like `protocol_type`, `service`, and `flag` are label-encoded using the `label_encoders` created during training.
   - Standard scaling is applied to normalize the features using the `scaler` fitted during training.

- **Neural Network Architecture:**
   - The model architecture includes an input layer, one hidden layer with 16 neurons using the 'relu' activation function, and an output layer with 1 neuron using the 'sigmoid' activation function.

- **Model Compilation:**
   - The model is compiled with the binary crossentropy loss function, the Adam optimizer with a learning rate of 0.001, and the accuracy metric.

- **Training Execution:**
   - The model is trained for 100 epochs on the training dataset (`X_train`, `y_train`) using the `fit` method.


### Evaluation Metrics

After training, the model's performance is evaluated using the test dataset (`X_test`, `y_test`). The following metrics are used for evaluation:

#### Accuracy
- The accuracy metric represents the ratio of correctly predicted instances to the total instances in the test set.
- It provides a general overview of the model's correctness.

#### Confusion Matrix
- A confusion matrix is a table that summarizes the performance of a classification model.
- It shows the number of true positives, true negatives, false positives, and false negatives.
- Metrics derived from the confusion matrix include precision, recall, and F1 score, providing insights into the model's ability to correctly classify instances and handle anomalies.

#### Output of Confusion Matrix:
- True Positives (TP): 4014 instances were correctly classified as anomalies.
- True Negatives (TN): 3498 instances were correctly classified as normal.
- False Positives (FP): 18 instances were incorrectly classified as anomalies.
- False Negatives (FN): 28 instances were incorrectly classified as normal.

#### Model Performance:
- Accuracy: 99.39%
- Loss: 0.0374
The high accuracy and low loss indicate that the model performed exceptionally well on the test set. The confusion matrix reveals a small number of misclassifications (18 false positives and 28 false negatives), demonstrating the model's strong ability to distinguish between normal and anomalous instances. 
## Model Predictions

After successfully training and evaluating our neural network model, we can leverage it to make predictions on new or unseen data. The process involves several key steps:

In this case raw data lacks the target variable and mirrors the order of our training data, we demonstrate the procedure for making predictions using our trained neural network.

#### Raw Data Transformation:

- We initialize raw_data, representing the new, unseen instance. This raw input is converted into a pandas DataFrame (test_data) with column names matching all feature names except 'class.'

#### Preprocessing Steps:

To maintain consistency with our training process, we apply identical preprocessing steps. This encompasses label encoding for categorical features (protocol_type, service, and flag) and scaling using the previously fitted scaler.

#### Scaling Features:

Features in the new instance (test_data) are scaled using the same scaler fitted during training.

#### Making a Prediction:

Our trained neural network model (model_tf) is employed to predict outcomes on the preprocessed new data (test_data_scaled).

#### Interpreting Prediction:

The predicted class is interpreted, and the result is presented accordingly.
This process exemplifies our model's capacity to make effective predictions on new data, offering insights into the classification of network activities as either "Normal" or "Anomaly."







## Improvement Opportunities

To optimize our Intrusion Detection System, consider the following improvements:

#### Hyperparameter Tuning:

- Fine-tune neural network parameters for enhanced performance.
#### Feature Engineering:

- Explore additional features for improved discrimination.
#### Ensemble Methods:

- Combine predictions from multiple models for increased robustness.
#### Advanced Architectures:

- Experiment with sophisticated neural network structures (e.g., RNNs, LSTMs).

#### Anomaly Detection Algorithms:

- Integrate other anomaly detection methods to complement the neural network.
#### Continuous Monitoring:

- Establish a system for periodic model retraining with new data.

#### User Feedback Integration:

- Incorporate user feedback into the training process for iterative refinement.

These enhancements aim to optimize the model's performance, adaptability, and real-world effectiveness. Continual refinement is essential for ensuring the system stays effective and relevant over time.

Your valuable contributions are highly encouraged in this project. Whether you have innovative ideas for enhancements or you spot any issues that need fixing, don't hesitate to open an issue or submit a pull request.

I trust that this project serves as a source of inspiration, igniting your curiosity to explore the boundless potential of Deep Learning and predictive modeling. Happy Coding!