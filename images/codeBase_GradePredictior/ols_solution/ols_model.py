import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.preprocessing import StandardScaler


docs_folder = '../../documentation/ols_docs/'

# Step 1️: Load the training and test datasets
train_data = pd.read_csv("../../data/train/train_data.csv")
test_data = pd.read_csv("../../data/validate/test_data.csv")

# Step 2️: Define input features and target variable for training & testing
X_train = train_data[["Socioeconomic Score", "Study Hours", "Sleep Hours", "Attendance (%)"]]
y_train = train_data["Grades"]
X_test = test_data[["Socioeconomic Score", "Study Hours", "Sleep Hours", "Attendance (%)"]]
y_test = test_data["Grades"]

# Step 3️: Normalize the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Scale training data
X_test_scaled = scaler.transform(X_test)  # Scale test data

# Step 4️: OLS requires a DataFrame with column names
X_train_ols = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_train_ols = sm.add_constant(X_train_ols)  # Add constant term (intercept)

X_test_ols = pd.DataFrame(X_test_scaled, columns=X_test.columns)
X_test_ols = sm.add_constant(X_test_ols)  # Add constant term (intercept)

# Step 5️: Train the OLS model with the normalized data
ols_model = sm.OLS(y_train, X_train_ols).fit()

# Step 6️: Print and save the model summary
with open(os.path.join(docs_folder, "ols_summary.txt"), "w") as file:
    file.write(str(ols_model.summary()))
print(ols_model.summary())

# 7️ Save the trained OLS model as a `.pkl` file
with open("currentOlsSolution.pkl", "wb") as file:
    pickle.dump(ols_model, file)

# Step 8️: Visualizations (Scatter Plot & Residual Plot)
plt.figure(figsize=(8,6))
plt.scatter(y_test, ols_model.predict(X_test_ols), alpha=0.5)
plt.xlabel("Actual Grades")  # Updated label
plt.ylabel("Predicted Grades")  # Updated label
plt.title("OLS Model: Actual vs. Predicted Grades")  # Updated title
plt.savefig(os.path.join(docs_folder,"scatter_plot.png"))
plt.show()

residuals = y_test - ols_model.predict(X_test_ols)
sns.histplot(residuals, bins=30, kde=True)
plt.xlabel("Residuals")  # Updated label
plt.ylabel("Count")  # Updated label
plt.title("Histogram of Residuals (Prediction Errors)")  # Updated title
plt.savefig(os.path.join(docs_folder, "residual_plot.png"))
plt.show()
