# --------- Dim Reduction (Feature extraction): Linear Discriminant Analysis (LDA) ---------
# Library
import pandas as pd
import matplotlib.pyplot as pLt
import numpy as np

# Data Extract
dataSet = pd.read_csv("Wine.csv")
X = dataSet.iloc[:, :13].values
y = dataSet.iloc[:, 13].values


# Data Split
from sklearn.model_selection import train_test_split
# 0.20 test_size means "1/5"th of the total observation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20, random_state = 0)

# Feature-Scaling
from sklearn.preprocessing import StandardScaler
#  y need not to be scaled.
st_x= StandardScaler()    
X_train= st_x.fit_transform(X_train)    
X_test= st_x.transform(X_test)  

# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# dcmPose_lda = LinearDiscriminantAnalysis(n_components= None)
dcmPose_lda = LinearDiscriminantAnalysis(n_components= 2)
# notice in LDA "fit_transform" takes both independent & dependent: X_train, y_train
X_train = dcmPose_lda.fit_transform(X_train, y_train)
X_test = dcmPose_lda.transform(X_test)
# explained_variance = dcmPose_lda.explained_variance_ratio_

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
              alpha = 0.30, cmap = ListedColormap(('red', 'green', 'orange')))
pLt.xlim(X1.min(), X1.max())
pLt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    pLt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'orange'))(i), label = j)
pLt.title('Logistic Regression (Training set)')
pLt.xlabel('LDA_1')
pLt.ylabel('LDA_2')
pLt.legend()
pLt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
pLt.contourf(X1, X2, classifer.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
              alpha = 0.30, cmap = ListedColormap(('red', 'green', 'orange')))
pLt.xlim(X1.min(), X1.max())
pLt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    pLt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'orange'))(i), label = j)
pLt.title('Logistic Regression (Test set)')
pLt.xlabel('LDA_1')
pLt.ylabel('LDA_2')
pLt.legend()
pLt.show()


# python prctc_logistic_rgsn_pca.py