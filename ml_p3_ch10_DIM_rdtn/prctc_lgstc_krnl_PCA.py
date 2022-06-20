# Library
import pandas as pd
import matplotlib.pyplot as pLt
import numpy as np

# Data Extract
dataSet = pd.read_csv("Social_Network_Ads.csv")
X = dataSet.iloc[:, [2,3]].values
y = dataSet.iloc[:, 4].values


# Data Split
from sklearn.model_selection import train_test_split
# 0.25 test_size means "1/4"th of the total observation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state = 0)

# Feature-Scaling
from sklearn.preprocessing import StandardScaler 
st_x= StandardScaler()    
X_train= st_x.fit_transform(X_train)    
X_test= st_x.transform(X_test)  


# applying Kernel-PCA : Unsupervised
from sklearn.decomposition import KernelPCA
dcomposer_kr_PCA = KernelPCA(n_components= 2, kernel="rbf")
X_train = dcomposer_kr_PCA.fit_transform(X_train)
X_test = dcomposer_kr_PCA.transform(X_test)


# Fit dataset to Logistic regression
from sklearn.linear_model import LogisticRegression # import class
# instead of "regressor" we now use "classifier"
classifer = LogisticRegression(random_state= 0) # create object
classifer.fit(X_train, y_train) # fit the dataset

# Predict
y_prd = classifer.predict(X_test)

# Making the confusion matrix use the function "confusion_matrix"
# Class in capital letters, functions are small letters 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true= y_test, y_pred= y_prd)
# parameters of cm: y_true: Real values, y_pred: Predicted value



# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
pLt.contourf(X1, X2, classifer.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.30, cmap = ListedColormap(('red', 'green')))
pLt.xlim(X1.min(), X1.max())
pLt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    pLt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
pLt.title('Logistic Regression applying Kernel-PCA (Training set)')
pLt.xlabel('PC_1 : ("Age" after kernel-PCA)')
pLt.ylabel('PC_2 : "Estimated Salary" after kernel-PCA')
pLt.legend()
pLt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
pLt.contourf(X1, X2, classifer.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.30, cmap = ListedColormap(('red', 'green')))
pLt.xlim(X1.min(), X1.max())
pLt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    pLt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
pLt.title('Logistic Regression applying Kernel-PCA (Test set)')
pLt.xlabel('PC_1 : ("Age" after kernel-PCA)')
pLt.ylabel('PC_2 : "Estimated Salary" after kernel-PCA')
pLt.legend()
pLt.show()

# python prctc_lgstc_krnl_PCA.py

