
import pandas as pd
import os
import seaborn
import pandas as pd
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


docs_folder = '../../documentation/ai/'

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
print(prediction)

data_for_heatmap = pd.DataFrame(input_test, columns=["Socioeconomic Score", "Study Hours", "Sleep Hours", "Attendance (%)"])
data_for_heatmap["Predicted Grades"] = output_pred.flatten()
data_for_heatmap["Actual Grades"] = output_test.values

# Visualize the results
plt.figure(figsize=(10, 8))
seaborn.heatmap(data_for_heatmap.corr(), annot=True, cmap='coolwarm')
plt.title('Korrelation zwischen Eingaben und Ergebnissen')
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
