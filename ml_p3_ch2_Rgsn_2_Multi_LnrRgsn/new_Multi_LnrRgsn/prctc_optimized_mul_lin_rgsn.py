# After Backward Elimination Feature selection
# Building Multiple Linear Regression model by only using R&D spend
import matplotlib as plt
import pandas as pd
import numpy as np

# import dataset
dataSet = pd.read_csv("50_Startups.csv")
X_bak_eli = dataSet.iloc[:, 0].values # R&D spend
y_bak_eli = dataSet.iloc[:, 4].values # 5th row


#converting Dataframe to Vector/Array
X_go = np.array(X_bak_eli) 
y_go = np.array(y_bak_eli) 


# split dataset to Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_go, y_go, test_size = 0.2, random_state = 0)


# Fitting multiple linear regression on traing set
from sklearn.linear_model import LinearRegression
regResor = LinearRegression()
# Reshape your data either using array.reshape(-1, 1) if your data has a single feature
regResor.fit(X_train.reshape(-1, 1), y_train)

# predict on the test-set X_test
y_pred = regResor.predict(X_test.reshape(-1, 1))


#Checking the score  
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
print('\n\n------------ Train Score: ', regResor.score(X_train.reshape(-1, 1), y_train))  
print('\n\n------------ Test Score: ', regResor.score(X_test.reshape(-1, 1), y_test)) 

# python prctc_optimized_mul_lin_rgsn.py