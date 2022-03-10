import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------- data preprocessing ------------------
# Importing the dataset. previous problem of "Bluff Detection"
datASet = pd.read_csv("Position_Salaries.csv")
X = datASet.iloc[:, 1:2].values
y = datASet.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set: No need here

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y.reshape(-1, 1))

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
# regressor.fit(X, y)
regressor.fit(X_scaled, y_scaled)

# ---------------- Visualising the SVR results ----------------
# Feature scaling is needed
# plt.scatter(X, y, color = 'red')
# plt.plot(X, regressor.predict(X), color = 'blue')

plt.scatter(X_scaled, y_scaled, color = "red")
plt.plot(X_scaled, regressor.predict(X_scaled), color = "blue")
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X_scaled), max(X_scaled), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_scaled, y_scaled, color = "red")
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



# ------------ prediction -----------
"""
Here we will predict the output for level 6.5 
because the candidate has 4+ years' experience as a regional manager, 
so he must be somewhere between levels 7 and 6.
"""
# y_prd = regressor.predict([[6.5]])  # [[6.5]], since Parmeter must be 2-D array

# We need to transfom 6.5 in our scaling
# y_prd = regressor.predict(sc_X.transform([[6.5]])) # alternative
y_prd = regressor.predict(sc_X.transform(np.array([[6.5]])))
print("prediction under scaled data", y_prd)
y_inv_sc = sc_y.inverse_transform(y_prd)
print("Reverse scaled prediction", y_inv_sc)

