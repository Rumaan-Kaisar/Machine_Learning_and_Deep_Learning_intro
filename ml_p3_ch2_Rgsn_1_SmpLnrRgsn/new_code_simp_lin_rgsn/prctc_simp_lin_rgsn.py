# ------------------- Importing the libraries -------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ------------------- Importing the dataset -------------------
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# ------------------- Splitting the dataset into the Training set and Test set -------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# -------------------  linear Regression ---------------------
from sklearn.linear_model import LinearRegression   # import library
s_l_regressor = LinearRegression()                  # regressor object
s_l_regressor.fit(X_train, y_train)                 # fit train data


# Predicting the test set results. Test set will be used
y_pred = s_l_regressor.predict(X_test)


# visualising the Training-set result
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, s_l_regressor.predict(X_train), color = "blue")   # notice the Train set is used
plt.title("Salary vs Experience 'Trainig Set'")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# visualising the Test-set result
plt.scatter(X_test, y_test, color = "red")
plt.plot(X_train, s_l_regressor.predict(X_train), color = "blue")   # notice the Train set is used again
plt.title("Salary vs Experience 'Trainig Set'")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# python prctc_simp_lin_rgsn.py


