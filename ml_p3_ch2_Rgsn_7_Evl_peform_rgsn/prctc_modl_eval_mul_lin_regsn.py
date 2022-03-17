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

# ---------- Only P-value implement automatic Backward Elimination: No manual iteration is needed ----------------
"""
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

"""

import statsmodels.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50, 6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(endog = y_go, exog=x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    
    regressor_OLS.summary()
    return x


SL = 0.05
X_pre_opt = np.append(arr = np.ones(shape = (50, 1)).astype(int), values =X_go , axis =1) # interchange the columns
X_opt = X_pre_opt[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

# python prctc_modl_eval_mul_lin_regsn.py

