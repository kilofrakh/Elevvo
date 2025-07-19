import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt


data =pd.read_csv('data.csv')

data = data.dropna()  

print(data.info())
print(data.describe())

z1 = data['Hours_Studied']
z2 = data['Exam_Score']

plt.scatter(z1, z2)
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.show()

