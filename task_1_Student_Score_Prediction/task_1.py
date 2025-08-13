import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# Data loading and cleaning
data = pd.read_csv('Student_Score_Prediction.csv')
data.dropna(inplace=True)  # Handle missing values

# Exploratory Data Analysis (EDA)
print(data.info())
print(data.describe())
print(data.dtypes)
print(data.head())
print(data.columns)

# Handle categorical data (if any)
catg = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=catg)

# Data visualization
plt.scatter(data['Hours_Studied'], data['Exam_Score'])
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Study Hours vs Exam Score')
plt.show()

# Feature and target split
# Assuming 'Hours_Studied' is the feature and 'Exam_Score' is the target
x = data[['Hours_Studied']]  # Explicitly select feature(s)
y = data['Exam_Score']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Linear Regression
model1 = LinearRegression()
model1.fit(x_train, y_train)
y_pred = model1.predict(x_test)

# Plot for Linear Regression
plt.scatter(y_test, y_pred, label='Predictions', alpha=0.5)
plt.xlabel('Actual Exam Scores')
plt.ylabel('Predicted Exam Scores')
plt.title('Actual vs Predicted Exam Scores (Linear Regression)')
plt.legend()
plt.show()

# Metrics for Linear Regression
print("Linear Regression Metrics:")
print(f'MSE: {mean_squared_error(y_test, y_pred):.4f}')
print(f'R²: {r2_score(y_test, y_pred):.4f}')
print(f'MAE: {mean_absolute_error(y_test, y_pred):.4f}\n')

# Polynomial Regression (try degree=2 or higher)
poly = PolynomialFeatures(degree=2)  # Changed to degree=2
x_poly_train = poly.fit_transform(x_train)
x_poly_test = poly.transform(x_test)

poly_model = LinearRegression()
poly_model.fit(x_poly_train, y_train)
y_poly_pred = poly_model.predict(x_poly_test)

# Plot for Polynomial Regression
plt.scatter(y_test, y_poly_pred, label='Predictions', alpha=0.5)
plt.xlabel('Actual Exam Scores')
plt.ylabel('Predicted Exam Scores')
plt.title('Actual vs Predicted Exam Scores (Polynomial Regression)')
plt.legend()
plt.show()

# Metrics for Polynomial Regression
print("Polynomial Regression Metrics:")
print(f'MSE: {mean_squared_error(y_test, y_poly_pred):.4f}')
print(f'R²: {r2_score(y_test, y_poly_pred):.4f}')
print(f'MAE: {mean_absolute_error(y_test, y_poly_pred):.4f}')

# Comparison of Linear and Polynomial Regression
plt.scatter(y_test, y_pred, label='Linear Regression', alpha=0.5)
plt.scatter(y_test, y_poly_pred, label='Polynomial Regression', alpha=0.5)
plt.xlabel('Actual Exam Scores')
plt.ylabel('Predicted Exam Scores')
plt.title('Comparison of Linear and Polynomial Regression Predictions')
plt.legend()
plt.show()