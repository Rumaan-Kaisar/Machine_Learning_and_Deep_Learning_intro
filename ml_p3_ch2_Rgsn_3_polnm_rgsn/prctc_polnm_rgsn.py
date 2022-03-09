# --------------- Import Libraries -------------- 
import pandas as pNd
import matplotlib.pyplot as pLt
import numpy as nPy



# --------------- Data Preprocessing  -------------
dataSet = pNd.read_csv("Position_Salaries.csv")
X = dataSet.iloc[:, 1:2].values
# X = dataSet.iloc[:, [1]].values # same as above
y = dataSet.iloc[:, 2].values



# ---------- Simple Linear model for Referace -------------
from sklearn.linear_model import LinearRegression
lin_regressor_1 = LinearRegression()    # object for Simple Linear Regression
lin_regressor_1.fit(X, y)



# ----------- Polynomial regressor --------------------
from sklearn.preprocessing import PolynomialFeatures
# Set the polynomial for X
poly_regressor = PolynomialFeatures(degree= 4)
X_poly = poly_regressor.fit_transform(X) # Generates Polynomial Feature-Matrix
lin_regressor_2 = LinearRegression()    # object for Polinomial Regression
lin_regressor_2.fit(X_poly, y)



# ------------  Visualize the result  --------------
    # ======= Simple  Linear ============
y_lin_pred = lin_regressor_1.predict(X)
pLt.scatter(X, y, color = "red")
pLt.plot(X, y_lin_pred, color = "blue")
pLt.title('Truth or bluff Linear Regression')
pLt.xlabel('Position Level')
pLt.ylabel("Salary")
pLt.show()



    # ======= Polynomial ============
    
# y_poly_pred = lin_regressor_2.predict(X): Does not work- Have to use Polynomial Feature-Matrix
y_poly_pred = lin_regressor_2.predict(X_poly)
pLt.scatter(X, y, color = "red")
pLt.title('Truth or bluff Linear Regression')
pLt.xlabel('Position Level')
pLt.ylabel("Salary")
pLt.plot(X, y_poly_pred, color = "green")
pLt.show()


# ----------------- Prediction -------------------
lin_pred = lin_regressor_1.predict([[6.5]])  # Simple linear Regerssion
print(lin_pred)  
poly_pred = lin_regressor_2.predict(poly_regressor.fit_transform([[6.5]]))  # Polynomial linear Regerssion
print(poly_pred)  



# python prctc_polnm_rgsn.py