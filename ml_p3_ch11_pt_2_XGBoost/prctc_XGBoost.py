
# ----------------- XGBoost instead of ANN -----------------


# Library
import pandas as pd
import matplotlib.pyplot as pLt
import numpy as np



# ----------------------------------- Part 1 : Data Preprocessing -----------------------------------------
# Data Extract
dataSet = pd.read_csv("Churn_Modelling.csv")
# X = dataSet.iloc[:, 3:-1].values # this can be used too
X = dataSet.iloc[:, 3:13].values # all columns from index 3, excluding 13 indexed column
y = dataSet.iloc[:, 13].values # the last column


# ------------- Encode Categorical Data  ----------- 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

#Encode "Gender" using LabelEncoder."Gender" is in "3rd-column", hence X[:, 2]
label_encode = LabelEncoder()
# Following is applicable to numpy array, if we used "X = dataSet.iloc[:, 3:13]"
# X = np.array(X) # it is needed if X is not an Arry. i.e. ".values" not applied
X[:, 2] = label_encode.fit_transform(X[:, 2])
print(X)

# For a data-set we can still encode it using Columns "key"
# X["Gender"] = label_encode.fit_transform(X["Gender"])


#Encode 'Gegraohy' using OneHotEncoder. "Gegraohy" is in "2nd-column", hence [1]
ct = ColumnTransformer(transformers = [("encoding", OneHotEncoder(), [1])], remainder = 'passthrough')
# remainder = 'passthrough' for remaining columns to be unchanged
X = ct.fit_transform(X) 
X = np.array(X) # convert this output to NumPy array
print(X)
X = X[:, 1:] # Excluding 0-index column to avoid dummy-variable trap



# ------------------ Data Split -----------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 0)


# ----------------------------------- Part 2 :Fitting XGBoost in the training set-----------------------------------
from xgboost import XGBClassifier
claSifire_XGB = XGBClassifier()
claSifire_XGB.fit(X_train, y_train)


# ----------------------------------- Part 3 : Predictions and Evaluating the model --------------------------------

# Predict
y_prd = claSifire_XGB.predict(X_test)


# Making the confusion matrix use the function "confusion_matrix"
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true= y_test, y_pred= y_prd)
print(f"\nConfusion Matrix :\n {cm}")
# parameters of cm: y_true: Real values, y_pred: Predicted value

accURacy_by_Confusion_matrix = (cm[0][0] + cm[1][1])/X_test.shape[0]
print(f"\nAccuracy = {accURacy_by_Confusion_matrix}%")


# --------   K-folds cross-validation  -------------
from sklearn.model_selection import cross_val_score
accuRacies = cross_val_score(estimator=claSifire_XGB, X = X_train, y = y_train, cv= 10)
mean_accu = accuRacies.mean()
std_accu = accuRacies.std()

print("\nAccuracy using k-fold-cross-validation: {:.2f} %".format(mean_accu*100))
print("\nStandard Deviation from k-fold-cross-validation: {:.2f} %".format(std_accu*100))

# python prctc_XGBoost.py