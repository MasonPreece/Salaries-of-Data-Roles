#Importing libraries into code
import kagglehub
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Downloading dataset
path = kagglehub.dataset_download("arnabchaki/data-science-salaries-2023")

print("Path to dataset files:", path)

#reading dataset
ds = pd.read_csv("ds_salaries.csv")

#Seeing info about dataset
print(ds.info())
print(" ")
print(ds.head())
print(" ")


# Group by region and calculate mean salary and printing it out
pay_across_regions = ds.groupby('employee_residence')['salary_in_usd'].mean().reset_index()
pay_across_regions = pay_across_regions.sort_values(by='salary_in_usd', ascending=False)
print("salary for regions")
print(pay_across_regions)

#analyzing pay by sorting job title and salary
pay_job_title = ds.groupby('job_title')['salary_in_usd'].mean().reset_index()
pay_job_title = pay_job_title.sort_values(by='salary_in_usd', ascending=False)
print("\nHighest to Lowest Pay for Job Title")
print(pay_job_title)

#analyzing pay based off of experience level and salary
pay_experience = ds.groupby('experience_level')['salary_in_usd'].mean().reset_index()
pay_experience = pay_experience.sort_values(by='salary_in_usd', ascending=False)
print("\nHighest to Lowest Pay for Job accoring to experience")
print(pay_experience)

# Drop unnecessary columns if any
ds = ds.drop(columns=['salary', 'salary_currency'], errors='ignore')  # 'salary' and 'salary_currency' may not be needed

# Check for missing values
print("Missing values:\n", ds.isnull().sum())

# Handle missing values (if any) - dropping rows with missing values for simplicity
ds = ds.dropna()

# Encode categorical variables
categorical_columns = ['job_title', 'employee_residence', 'company_location', 
                       'experience_level', 'company_size', 'remote_ratio']

# Use Label Encoding for simplicity and assign values in columns with numerical values
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    ds[col] = le.fit_transform(ds[col])
    label_encoders[col] = le

# Define feature matrix (X) and target variable (y). This will give a chart and x is the columns that will be measured
#y is the variable that we are trying to predict
X = ds[['job_title', 'experience_level', 'company_size', 'remote_ratio',
        'employee_residence', 'company_location']]
y = ds['salary_in_usd']

# Split the data into training and testing sets to figure out the target y and uses the dataset for prediction
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model 
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Display feature importance
import matplotlib.pyplot as plt

feature_importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, feature_importance)
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Salary Prediction')
plt.show()

