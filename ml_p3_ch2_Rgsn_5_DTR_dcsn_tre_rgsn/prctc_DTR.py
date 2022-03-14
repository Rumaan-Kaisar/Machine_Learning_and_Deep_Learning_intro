# Library
import pandas as pd
import matplotlib.pyplot as pLt
import numpy as np


# Data Extract
dataSet = pd.read_csv("Position_Salaries.csv")
X = dataSet.iloc[:, 1:2].values
y = dataSet.iloc[:, 2].values


# # Feature-Scaling
# from sklearn.preprocessing import StandardScaler
# sc_x = StandardScaler()
# sc_y = StandardScaler()
# X_scaled = sc_x.fit_transform(X)
# y_scaled = sc_y.fit_transform(y.reshape(-1, 1))


# Data Split : No need for this example

# Fit dataset to model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state= 0)
regressor.fit(X, y)


# Predict
y_pred = regressor.predict([[6.5]])
print("The predicte value ffor 6.5 is : ", y_pred)

# plot the model
# pLt.scatter(X, y, color = "red")
# pLt.plot(X, regressor.predict(X), color = "green")
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1) # reshape matrix/array
pLt.scatter(X, y, color = "red")
pLt.plot(X_grid, regressor.predict(X_grid), color = "green")
pLt.title("Truth or Bluff (Decision Tree Regression)")
pLt.xlabel("Position level")
pLt.ylabel("Salary")
pLt.show()


# python prctc_DTR.py