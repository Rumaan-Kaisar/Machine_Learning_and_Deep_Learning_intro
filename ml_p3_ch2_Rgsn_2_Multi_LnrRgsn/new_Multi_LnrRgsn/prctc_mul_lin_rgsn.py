# Fundamental  Libraries
import matplotlib as plt
import pandas as pd
import numpy as np

# import dataset
dataSet = pd.read_csv("50_Startups.csv")
X = dataSet.iloc[:, :-1] #all rows except last
y = dataSet.iloc[:, 4] # 5th row


# categorical to numerical
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
colTfrm = ColumnTransformer(transformers = [("encoder", OneHotEncoder(), [3])], remainder="passthrough" )
X_encoded = np.array(colTfrm.fit_transform(X))
        #  last column is now replaced with dummy colums (1st 3 colmns)

# Avoiding dummy-var trap: omit one dummy varable
X_go = X_encoded[:, 1:]         # select all columns starting from 2nd column
y_go = np.array(y) #converting Dataframe to Vector/Array


# split dataset to Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_go, y_go, test_size = 0.2, random_state = 0)


# Fitting multiple linear regression on traing set
from sklearn.linear_model import LinearRegression
regResor = LinearRegression()
regResor.fit(X_train, y_train)

# predict on the test-set X_test
y_pred = regResor.predict(X_test)


#Checking the score  
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
print('\n\n------------ Train Score: ', regResor.score(X_train, y_train))  
print('\n\n------------ Test Score: ', regResor.score(X_test, y_test)) 

#================= building the optimal model using Backward elimination ====================
"""
# ----------- step 1 : Preprosecc for OLS ------------
# import statsmodels.formula.api as smf --------------------- LEGACY CODE
import statsmodels.api as smf
# X_opt = np.append(arr = X_go, values = np.ones(shape = (50, 1)).astype(int), axis =1) 
# 50 for row and 1 for column (row, column)
# convert to int type
# set axis = 1: column 0:row
X_pre_opt = np.append(arr = np.ones(shape = (50, 1)).astype(int), values =X_go , axis =1) # interchange the columns


# --------- iteration 1 ------------

# ------------ step 2 : fit with OLS -------------------
X_opt = X_pre_opt[:, [0, 1, 2, 3, 4, 5]]  # new vector whiuch will  be optimized
# Lets we set SL = 0.05 explicitly. For our learning pupose
# Re-fit  with new regressor. Used OLS "Ordinary Least Squares"
regressor_OLS = smf.OLS(endog = y_go, exog=X_opt).fit() # fitting with OLS

# ------------ step 3 : inspect p-values -------------------
print(regressor_OLS.summary()) # to inspect the p-valuse

# --------- iteration 2 ------------
X_opt = X_pre_opt[:, [0, 1, 3, 4, 5]]  # removed 3rd column (x2 of iteration 1's X_opt)
regressor_OLS = smf.OLS(endog = y_go, exog=X_opt).fit() # Re-fit with OLS
print(regressor_OLS.summary()) # to inspect the p-valuse

# --------- iteration 3 ------------
X_opt = X_pre_opt[:, [0, 3, 4, 5]]  # removed 2nd column (x1 of iteration 2's X_opt)
regressor_OLS = smf.OLS(endog = y_go, exog=X_opt).fit() # Re-fit with OLS
print(regressor_OLS.summary()) # to inspect the p-valuse


# --------- iteration 4 (can be final iteartion) ------------
X_opt = X_pre_opt[:, [0, 3, 5]]  # removed 3rd column (x2 of iteration 3's X_opt)
regressor_OLS = smf.OLS(endog = y_go, exog=X_opt).fit() # Re-fit with OLS
print(regressor_OLS.summary()) # to inspect the p-valuse


# --------- iteration 5 (final iteartion) ------------
X_opt = X_pre_opt[:, [0, 3]]  # removed 3rd column (x2 of iteration 4's X_opt)
regressor_OLS = smf.OLS(endog = y_go, exog=X_opt).fit() # Re-fit with OLS
print(regressor_OLS.summary()) # to inspect the p-valuse

"""

# ---------- implement automatic Backward Elimination: No manual iteration is needed ----------------
import statsmodels.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(endog = y_go, exog=x).fit()
        # picking the max p-value
        maxPval = max(regressor_OLS.pvalues).astype(float)
        if maxPval >= sl:
            for j in range(0, numVars - i):
                # deleting the clumn
                if (regressor_OLS.pvalues[j].astype(float) == maxPval):
                    x = np.delete(x, j, 1) # deletes j-th column. "1" is used for "column". To delet "row" use "0"
    print(regressor_OLS.summary())
    return x


SL = 0.05
X_pre_opt = np.append(arr = np.ones(shape = (50, 1)).astype(int), values =X_go , axis =1) # interchange the columns
X_opt = X_pre_opt[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)


# python prctc_mul_lin_rgsn.py

