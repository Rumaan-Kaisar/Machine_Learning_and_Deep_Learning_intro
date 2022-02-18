# ------------------ Data Preprocessing --------------------



# ------------------- Importing the libraries ----------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# x = np.arange(0, 5, 0.1)  
# y = np.sin(x)  
# plt.plot(x, y)  
# plt.show()




#-------------------- importing the dataset --------------------------
dataset = pd.read_csv("Data.csv")
print(dataset)
# ":-1" is to all columns exluding last column
# ":" to select all rows
old_x = dataset.iloc[:, :-1]
old_y = dataset.iloc[:, 3]
print("Before\n",old_x)




# -----------------------  Taking care of missing data ----------------------
# import sklearn
# print(sklearn.__version__)

from sklearn.impute import SimpleImputer
imptr = SimpleImputer(missing_values= np.nan, strategy="mean") # "NaN" connot be used. Use np.nan
# fix the 2nd and 3rd column
impt = imptr.fit(old_x.iloc[:, 1:3])
print(impt) # it is a method, acts on feature matrix
old_x.iloc[:, 1:3] = impt.transform(old_x.iloc[:, 1:3])
print("After\n",old_x)

# ooooooooo For older version oooooooooo
# from sklearn.preprocessing import Imputer
# imptr = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)




# -------------- Categorical data --------------
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# label_encode_x = LabelEncoder()
# old_x.iloc[:, 0] = label_encode_x.fit_transform(old_x.iloc[:, 0])
# print(old_x)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

#Encode Country Column
ct = ColumnTransformer(transformers = [("encoding", OneHotEncoder(), [0])], remainder = 'passthrough')
# remainder = 'passthrough' for remaining columns to be unchanged
old_x = ct.fit_transform(old_x) # convert this output to NumPy array
print(old_x)
old_x = np.array(old_x)
print(old_x)

#Encode dependent vector Column
label_encode = LabelEncoder()
old_y = label_encode.fit_transform(old_y)
print(old_y)




# ---------------------- Feature Scaling ------------------------
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
old_x = sc.fit_transform(old_x)
print(old_x)



# ------------ Splitting the Dataset: Train and Test ----------------
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(old_x, old_y, test_size= 0.2, random_state = 0)
print(x_train)
print(y_train)
print(x_test)
print(y_test)




# # ^^^^^^^^^^ List slicing ^^^^^^^^^^^

# # Initialize list
# Lst = [50, 70, 30, 20, 90, 10, 50]

# # Display list
# print(Lst[::])

# # Index -1 represents the last element and -n represents the first element of the list(considering n as the length of the list). Lists can also be manipulated using negative indexes also.


# # Initialize list
# Lst = [50, 70, 30, 20, 90, 10, 50]
 
# # Display list
# print(Lst[-7::1])

# # Display list
# print(Lst[1:5])


# python prcts_dt_prep.py