from requests import request
from bs4 import BeautifulSoup
import pandas as pd

import seaborn
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

""" 
trainings_data = pd.read_csv('../data/train/train_data.csv')

input_train = trainings_data[["Socioeconomic Score", "Study Hours", "Sleep Hours", "Attendance (%)"]] 
output_train = trainings_data["Grades"]  

# Step 2: Build the neural network model
model = models.Sequential([
    layers.Dense(256, input_dim=4, activation='sigmoid'),
    layers.Dense(256, activation='sigmoid'), 
    layers.Dense(1)                                   
])

# Step 3: Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 4: Train the model
model.fit(input_train, output_train, epochs=100, batch_size=32)

model.save('currentSolutions.h5') """

model = load_model('currentSolutions.h5')

test_data = pd.read_csv('../data/validate/test_data.csv')
input_test = test_data[["Socioeconomic Score", "Study Hours", "Sleep Hours", "Attendance (%)"]] 
output_test = test_data["Grades"]  

# Predict using the trained model
output_pred = model.predict(input_test)

sample_input = pd.DataFrame([[0.5, 100, 8, 100]], columns=["Socioeconomic Score", "Study Hours", "Sleep Hours", "Attendance (%)"])
prediction = model.predict(sample_input)
print(prediction)

data_for_heatmap = pd.DataFrame(input_test, columns=["Socioeconomic Score", "Study Hours", "Sleep Hours", "Attendance (%)"])
data_for_heatmap["Predicted Grades"] = output_pred.flatten()
data_for_heatmap["Actual Grades"] = output_test.values

plt.figure(figsize=(10, 8))
seaborn.heatmap(data_for_heatmap.corr(), annot=True, cmap='coolwarm')
plt.title('Korrelation zwischen Eingaben und Ergebnissen')
plt.show()

""" # Step 6: Visualize the results
plt.scatter(input_test, output_test, label='Data Points', color='blue')
plt.scatter(input_test, y_pred, label='Predicted by Neural Network', color='red', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Function Approximation with Neural Network')
plt.show() """

# Step 7: Evaluate the model (optional)
loss = model.evaluate(input_test, output_test)
print(f"Test Loss: {loss}")

""" 
# Create subplots (2 rows, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(10, 8))  # 2 rows, 2 columns

# Plot each subplot
axes[0].scatter(df["x"], df["y"], label="Data Points", color="blue")
axes[0].plot(df["x"], ols_predicted_y, color="red", label="Regression Line")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_title("OLS Regression Line")

axes[1].scatter(X_test, y_test, label='Data Points', color='blue')
axes[1].scatter(X_test, y_pred, label='Predicted by Neural Network', color='red', linestyle='--')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].legend()
axes[1].set_title('Function Approximation with Neural Network')


# Adjust layout to avoid overlapping
plt.tight_layout()
plt.show() """