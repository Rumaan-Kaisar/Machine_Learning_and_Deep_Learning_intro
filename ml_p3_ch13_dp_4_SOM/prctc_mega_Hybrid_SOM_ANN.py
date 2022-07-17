# --------- mega case study : Make a Hybrid Deep learning model ---------- 

# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -------------- Data Preprocessing -------------------
# importing the Data-set
dataset = pd.read_csv("Credit_Card_Applications.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#feature scaling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range= (0, 1))
X = sc.fit_transform(X)



# ------ Part 1 : Identify the frauds using Self Organizing Maps (SOM) ---------
# Training the SOM
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)


# Visualize the result
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()

marKers = ['o', 's']
coLors = ['r', 'g']
for i, x in enumerate(X):
    w= som.winner(x)
    plot(
        w[0]+0.5,
        w[1]+0.5,
        marKers[y[i]],
        markeredgecolor = coLors[y[i]],
        markerfacecolor = 'None',
        markersize = 10,
        markeredgewidth = 2)
show()

# finding the Fruds
mappins = som.win_map(X) # not 'x' its our dataset "X", capital X
fraud_list_1 = mappins[(5, 5)]
fraud_list_2 = mappins[(6, 4)]
frauDs = np.concatenate((mappins[(5, 5)], mappins[(6, 4)]), axis=0)
frauDs = sc.inverse_transform(frauDs)

# saving the list to a csv
# save numpy array as csv file
"""
from numpy import asarray
from numpy import savetxt
# define data
data = frauDs
# save to csv file
savetxt('frauds.csv', data, delimiter=',')
"""


# --------- part 2: Going from Unsupervised to Supervised Deep Learning ---------

# creating the matrix of features
cuStomers = dataset.iloc[:, 1:].values


# creating the dependent variable
is_fraud = np.zeros(len(dataset)) # creating 690 size vector of zeros
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauDs:
        is_fraud[i]=1


# creating ANN model
# ------------------ Feature-Scaling ----------------------
from sklearn.preprocessing import StandardScaler
#  y dependent variable, need not to be scaled: categorical variable, 0 and 1 
st_x= StandardScaler()    
cuStomers= st_x.fit_transform(cuStomers)    # replace "X_train" by 'cuStomers'




# -----------------------------------  Creating ANN model ---------------------------------------------

    # 1. importing "keras" libraries and packages
# from tensorflow import keras
import keras # using TensorFlow backend
from keras.models import Sequential
from keras.layers import Dense
    
    # 2. initialize the ANN
ann_classifier = Sequential()

    # 3. Add the "input-layer" and  "first Hidden-layer"
# ann_classifier.add(Dense(output_dim = 6, init = "uniform", activation = "relu", input_dim = 11))
ann_classifier.add(Dense(units = 2, kernel_initializer = "uniform", activation = "relu", input_dim = 15))

    # 4. Add the "second Hidden-layer"
# ann_classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))

    # 5. Add the "output-layer"
ann_classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))

    # 6. Compile the ANN
ann_classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics= ["accuracy"])

    # 7. Train the model: fit the ANN to Training-set (batch_size and epoch) 
ann_classifier.fit(cuStomers, is_fraud, batch_size= 1, epochs= 2)




# ----------------------------------- Part 3 : Predictions  ---------------------------------------------

# Predicting the probabilities of frauds
y_prd = ann_classifier.predict(cuStomers)

# concatenate columns to add customer_id
y_prd = np.concatenate((dataset.iloc[:, 0:1].values, y_prd), axis=1)

# sorting trick
y_prd = y_prd[y_prd[:, 1].argsort()] # y_prd[:, 1] selects the 2nd column of y_pred


# python prctc_mega_Hybrid_SOM_ANN.py