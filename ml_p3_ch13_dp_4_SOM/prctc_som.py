# Self Organizing Maps (SOM)

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
fraud_list_1 = mappins[(8, 6)]
fraud_list_2 = mappins[(1, 7)]
frauDs = np.concatenate((mappins[(8, 6)], mappins[(1, 7)]), axis=0)
frauDs = sc.inverse_transform(frauDs)



# python prctc_som.py