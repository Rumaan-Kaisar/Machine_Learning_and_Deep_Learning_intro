# NLP: Natural Language Precessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -------------- Data preprocessing --------------
# importing dataset
        # since we are using "tsv" ins tead of "csv" we need to specify some parameters.
        # Because "Pandas" expecting some "csv" files
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)


# Cleaning the text
import re
import nltk
# nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# print(f"{dataset['Review'][0]}")
coRpus_reViw = []

for i in range(0, 1000):
    rev_W = re.sub("[^a-zA-Z]", " ", dataset["Review"][i])
    rev_W = rev_W.lower()   # converting to Lower case
    rev_W = rev_W.split()   # converting "string" to "list"
    # reView = [wrd for wrd in rev_W if wrd not in stopwords.words("english")]
    # Removing stopwords. And stemming : Finding the root of different versions of a same word 
    prt_stmr = PorterStemmer()
    reView = [prt_stmr.stem(wrd) for wrd in rev_W if wrd not in set(stopwords.words("english"))]
    reVw = " ".join(reView)
    coRpus_reViw.append(reVw)

# creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
# cntVctzr = CountVectorizer()
cntVctzr = CountVectorizer(max_features= 1500)
X = cntVctzr.fit_transform(coRpus_reViw).toarray()
y = dataset.iloc[:, 1].values


# using ---------------- Naïve Bayes ---------------
# Data Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20, random_state = 0)

# # Feature-Scaling
# from sklearn.preprocessing import StandardScaler

# st_x= StandardScaler()    
# X_train= st_x.fit_transform(X_train)    
# X_test= st_x.transform(X_test)  


# Fit train set to Naïve Bayes classifier: No parmeter is needed
from sklearn.naive_bayes import GaussianNB
clsFier = GaussianNB()  
clsFier.fit(X_train, y_train) # fit the dataset

# Predict
y_prd = clsFier.predict(X_test)

# Making the confusion matrix use the function "confusion_matrix"
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true= y_test, y_pred= y_prd)
# parameters of cm: y_true: Real values, y_pred: Predicted value

accuracy = (cm[0][0] + cm[1][1])/X_test.shape[0]

# How can I find the length of a row (or column) of this matrix? Equivalently, how can I know the number of rows or columns?

# shape is a property of both numpy ndarray's and matrices.
    # A.shape
# will return a tuple (m, n), where m is the number of rows, and n is the number of columns.



# import nltk
# from nltk.corpus import stopwords
# print(stopwords.words('english'))
# print(set(stopwords.words('english')))

# python prctc_nlp.py