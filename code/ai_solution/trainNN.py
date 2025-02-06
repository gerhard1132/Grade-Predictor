
import pandas as pd
import os
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import matplotlib.pyplot as plt


docs_folder = '../../documentation/ai_docs/'

# Training process
trainings_data = pd.read_csv('../../data/train/train_data.csv')

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
history = model.fit(input_train, output_train, epochs=100, batch_size=32)

model.save('currentAiSolutions.h5')

# Getting already trained model
# model = load_model('currentAiSolutions.h5')

test_data = pd.read_csv('../../data/validate/test_data.csv')
input_test = test_data[["Socioeconomic Score", "Study Hours", "Sleep Hours", "Attendance (%)"]] 
output_test = test_data["Grades"]  

# Predict using the trained model
output_pred = model.predict(input_test)

sample_input = pd.DataFrame([[0.5, 100, 8, 100]], columns=["Socioeconomic Score", "Study Hours", "Sleep Hours", "Attendance (%)"])
prediction = model.predict(sample_input)
print(f'The models grade prediction is: {prediction}')

mae = mean_absolute_error(output_test, output_pred)
r2 = r2_score(output_test, output_pred)
rmse = np.sqrt(mean_squared_error(output_test, output_pred))

with open(os.path.join(docs_folder,'ai_summary.txt'), "w") as file:
    file.write(f"""
General metrics about the neural network:

Mean absolute Error (MAE): {mae}
R^2 Score: {r2}
Root Mean Squared Error (RMSE): {rmse}
""")

data_for_heatmap = pd.DataFrame(input_test, columns=["Socioeconomic Score", "Study Hours", "Sleep Hours", "Attendance (%)"])
data_for_heatmap["Predicted Grades"] = output_pred.flatten()
data_for_heatmap["Actual Grades"] = output_test.values

# Visualize the results
plt.figure(figsize=(10, 8))
sns.heatmap(data_for_heatmap.corr(), annot=True, cmap='coolwarm')
# Fix cropping issues
plt.xticks(rotation=45, ha="right")  
plt.yticks(rotation=0) 
plt.tight_layout()  

plt.title('Correlation between Inputs und Results')
plt.savefig(os.path.join(docs_folder,'ai_result_heatmap.png'))

# Step 7: Evaluate the model
loss = model.evaluate(input_test, output_test)
print(f"Test Loss: {loss}")

# Plot the loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(docs_folder,'loss_epochs.png'))

# Create a scatter plot to compare actual vs. predicted grades
plt.figure(figsize=(8, 6))
plt.scatter(output_test, output_pred, alpha=0.5, color="blue")
# Add a regression line
sns.regplot(x=output_test, y=output_pred, scatter=False, color="red", line_kws={"linewidth": 2}, label="Regression Line")
x_vals = np.linspace(min(output_test), max(output_test), 100)
plt.plot(x_vals, x_vals, color="green", linestyle="--", linewidth=2, label="Optimal Line")
plt.xlabel("Actual Grades")  
plt.ylabel("Predicted Grades")
plt.legend()
plt.title("AI Model: Actual vs. Predicted Grades")  

# Save the scatter plot in the documentation folder
plt.savefig(os.path.join(docs_folder, "ai_scatter_plot.png"))
plt.show()