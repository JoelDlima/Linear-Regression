import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

# Load the dataset
dataset = pd.read_csv('house_price_data.csv')

# Display missing values in each column
print("Missing values in each column:")
print(dataset.isnull().sum())

# Separate target variable from features
target_column = 'Price (INR in Lakhs)'
y = dataset[target_column]  # Target/output
X = dataset.drop(columns=[target_column])  # Input/features

# Identify categorical columns (if any)
categorical_cols = X.select_dtypes(include=['object']).columns
print("\nCategorical columns:", list(categorical_cols))

# One-hot encode categorical features (if present)
X = pd.get_dummies(X, drop_first=True)

# Replace missing numeric values with column mean
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split the dataset into training and test sets (80% train, 20% test)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Train a linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict on test set
pred_y = regressor.predict(x_test)

# Visualize training set results (works well only if one feature)
plt.scatter(x_train.iloc[:, 0], y_train)  # Plot using first feature
plt.plot(x_train.iloc[:, 0], regressor.predict(x_train), color='red')  # Regression line
plt.title('Price(INR) vs Sqft Area (Training Set)')
plt.xlabel('Sqft Area')
plt.ylabel('Price(INR in Lakhs)')
plt.show()

# Visualize test set results (same note as above)
plt.scatter(x_test.iloc[:, 0], y_test)
plt.plot(x_test.iloc[:, 0], regressor.predict(x_test), color='red')
plt.title('Price(INR) vs Sqft Area (Test Set)')
plt.xlabel('Sqft Area')
plt.ylabel('Price(INR in Lakhs)')
plt.show()

# Evaluate the model using MSE, RMSE, and R² score
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, pred_y)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred_y)
print(f"MSE: {mse}, RMSE: {rmse}, R²: {r2}")
