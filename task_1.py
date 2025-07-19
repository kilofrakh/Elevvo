import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# data cleaning
data = pd.read_csv('data.csv')
data.dropna(inplace=True)

print(data.info())
print(data.describe())
print(data.dtypes)
print(data.head())
print(data.columns)


catg = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=catg)

# Data visualization
plt.scatter(data['Hours_Studied'], data['Exam_Score'])
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Study Hours vs Exam Score')
plt.show()

# Feature and target split
x = data.drop(columns=['Exam_Score'])
y = data['Exam_Score']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Linear Regression
model1 = LinearRegression()
model1.fit(x_train, y_train)
y_pred = model1.predict(x_test)

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Exam Scores')
plt.ylabel('Predicted Exam Scores')
plt.title('Actual vs Predicted Exam Scores (Linear Regression)')
plt.show()

print(" Linear Regression Metrics:")
print(f'MSE: {mean_squared_error(y_test, y_pred):.4f}')
print(f'R²: {r2_score(y_test, y_pred):.4f}')
print(f'MAE: {mean_absolute_error(y_test, y_pred):.4f}\n')

# Polynomial Regression
poly = PolynomialFeatures(degree=1)
x_poly_train = poly.fit_transform(x_train)
x_poly_test = poly.transform(x_test)

poly_model = LinearRegression()
poly_model.fit(x_poly_train, y_train)
y_poly_pred = poly_model.predict(x_poly_test)

plt.scatter(y_test, y_poly_pred)
plt.xlabel('Actual Exam Scores')
plt.ylabel('Predicted Exam Scores')
plt.title('Actual vs Predicted Exam Scores (Polynomial Regression)')
plt.show()

print(" Polynomial Regression Metrics:")
print(f'MSE: {mean_squared_error(y_test, y_poly_pred):.4f}')
print(f'R²: {r2_score(y_test, y_poly_pred):.4f}')
print(f'MAE: {mean_absolute_error(y_test, y_poly_pred):.4f}')

# poly vs linear comparison
plt.scatter(y_test, y_pred, label='Linear Regression', alpha=0.5)
plt.scatter(y_test, y_poly_pred, label='Polynomial Regression', alpha=0.5)
plt.xlabel('Actual Exam Scores')
plt.ylabel('Predicted Exam Scores')
plt.title('Comparison of Linear and Polynomial Regression Predictions')
plt.legend()
plt.show()