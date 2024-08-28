# ML-ASSIGNMENT OMPAL
# Data Analysis and Modeling Examples

This repository contains examples of data analysis and modeling using Python. It includes code snippets for working with datasets, splitting data, training models, and evaluating performance. The examples use the Iris dataset and a sample dataset for linear regression.

## Prerequisites
To run this script, you need Python and the following libraries:

pandas for data manipulation
scikit-learn for machine learning tools
numpy for numerical operations

# Code Breakdown
# 1. Dataset Exploration
This section deals with loading and exploring the Iris dataset:

## Load the Iris Dataset:
Using scikit-learn's load_iris function.
Convert to DataFrame: Transform the dataset into a pandas DataFrame for easier manipulation.
## Display Initial Information:
Print the first five rows.
Print the shape of the dataset.
Print summary statistics (mean, std, min, max, etc.).
# 2. Data Splitting
This part involves splitting the Iris dataset into training and testing sets:

## Features and Target:
The Iris dataset features (X) and target (y) are separated.
## Train-Test Split: 
The dataset is split into training and testing sets with an 80-20 ratio.
## Print Sample Sizes:
Number of samples in each set is displayed.
# 3. Linear Regression
This section applies linear regression to a custom dataset:

## Custom Dataset Creation: 
A dataset with YearsExperience and Salary is created manually.
## Feature and Target Split: 
The dataset is divided into features (X) and target (y).
## Train-Test Split:
The custom dataset is split into training and testing sets.
## Model Training and Prediction:
Initialize and fit a Linear Regression model.
Predict salaries based on the test set features.
## Model Evaluation:
Calculate and print the Mean Squared Error (MSE) of the model.

### Code

```python


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# === Dataset Exploration ===
print("Dataset Exploration:")

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Display the first five rows
print("\nFirst five rows of Iris dataset:")
print(iris_df.head())

# Display the shape of the dataset
print("\nDataset shape:")
print(iris_df.shape)

# Display summary statistics
print("\nSummary statistics:")
print(iris_df.describe())

# === Data Splitting ===
print("\nData Splitting:")

# Split the Iris dataset into features and target
X = iris.data
y = iris.target

# Split the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the number of samples in both sets
print(f"Number of samples in training set: {X_train.shape[0]}")
print(f"Number of samples in testing set: {X_test.shape[0]}")

# === Linear Regression ===
print("\nLinear Regression:")

# Example dataset with YearsExperience and Salary
data = {
    'YearsExperience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [40000, 42000, 44000, 46000, 48000, 50000, 52000, 54000, 56000, 58000]
}

df = pd.DataFrame(data)

# Split into features and target
X = df[['YearsExperience']]
y = df['Salary']

# Split the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on the test set: {mse:.2f}")

