# Artificial Neural Network
# Install : Tensorflow, Keras and Theano libraries.

# Library
import pandas as pd
import matplotlib.pyplot as pLt
import numpy as np



# ----------------------------------- Part 1 : Data Preprocessing ---------------------------------------------
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

        # Feature-Scaling after Data Split



# ------------------ Feature-Scaling ----------------------
from sklearn.preprocessing import StandardScaler
#  y dependent variable, need not to be scaled: categorical variable, 0 and 1 
st_x= StandardScaler()    
X_train= st_x.fit_transform(X_train)    
X_test= st_x.transform(X_test)  



# ----------------------------------- Part 2 : Creating ANN model ---------------------------------------------

    # 1. importing "keras" libraries and packages
# from tensorflow import keras
import keras # using TensorFlow backend
from keras.models import Sequential
from keras.layers import Dense
    
    # 2. initialize the ANN
ann_classifier = Sequential()

    # 3. Add the "input-layer" and  "first Hidden-layer"
# ann_classifier.add(Dense(output_dim = 6, init = "uniform", activation = "relu", input_dim = 11))
ann_classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu", input_dim = 11))

    # 4. Add the "second Hidden-layer"
ann_classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))

    # 5. Add the "output-layer"
ann_classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))

    # 6. Compile the ANN
ann_classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics= ["accuracy"])

    # 7. Train the model: fit the ANN to Training-set (batch_size and epoch) 
ann_classifier.fit(X_train, y_train, batch_size= 10, epochs= 100)




# ----------------------------------- Part 3 : Predictions and Evaluating the model ---------------------------------------------

# Predict
y_prd = ann_classifier.predict(X_test)

# coverting probabilities into "true/false" form. because 1 for leaving the Bank
y_prd = (y_prd > 0.5)

# Making the confusion matrix use the function "confusion_matrix"
# Class in capital letters, functions are small letters 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true= y_test, y_pred= y_prd)
# parameters of cm: y_true: Real values, y_pred: Predicted value

accURacy = (cm[0][0] + cm[1][1])/X_test.shape[0]
print(f"Accuracy = {accURacy}%")


# python prctc_ANN.py