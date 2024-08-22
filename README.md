# Linear-Regression-Health-Costs-Calculator

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.read_csv('healthcare_costs.csv')

# Display the first few rows of the dataset
data.head()

# Convert categorical columns to numerical values
label_encoders = {}
categorical_columns = ['sex', 'smoker', 'region']

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Separate features and target variable
X = data.drop(columns=['expenses'])
y = data['expenses']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')

# Train the model
history = model.fit(X_train, y_train, epochs=20, validation_split=0.2)


# Evaluate the model
test_loss = model.evaluate(X_test, y_test)
print(f"Mean Absolute Error on Test Data: {test_loss}")

# Predict using the test dataset
y_pred = model.predict(X_test).flatten()

# Calculate mean absolute error
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Plot predictions vs true values
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions')
plt.plot([0, max(y_test)], [0, max(y_test)], color='red')  # Diagonal line
plt.show()
