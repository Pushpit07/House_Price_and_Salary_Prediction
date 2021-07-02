import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("Salary_Data.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

print('Predicted :', model.predict(X_test[:5]))
print('Actual :', y_test[:5])
print('Score :', model.score(X_test, y_test))


yrs = int(input("How many years of experience do you have? "))
sal = model.predict(np.array([yrs]).reshape(-1, 1))
print("\nYour estimated salary is $", round(sal[0], 3), "\n")