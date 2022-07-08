# Recurrent Neural Netwark: RNN

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ------------------- Data Preprocessing ---------------------
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')     # Notice train-set is now our main dataset
training_set = dataset_train.iloc[:, 1:2].values

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range= (0, 1))
train_set_scled = sc.fit_transform(training_set)


print(f"Size of train set : {train_set_scled.shape}")
print(f"No. of data-points : {train_set_scled.shape[0]}")


# creating a data structure with 60 time-steps and 1 output.
data_size = train_set_scled.shape[0]
X_train = []
y_train = []

for i in range(60, data_size):
    X_train.append(train_set_scled[(i-60):i, 0])    # append a list of 60 data-points in list of (data_size-60) lists
    y_train.append(train_set_scled[i, 0]) # vector of (data_size-60) rows

# Convert X_train and y_train to NumPy array
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping 
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# ------------------- Building RNN ---------------------

# importing keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
from keras.layers import Dropout

# Initialize the RNN
regressor_rnn = Sequential()

# adding first GRU layer & some Dropout-Regularization
regressor_rnn.add(GRU(units= 50, return_sequences= True, input_shape = (X_train.shape[1], 1)))
regressor_rnn.add(Dropout(rate = 0.2))

# second GRU layer with Dropout-Regularization
regressor_rnn.add(GRU(units= 50, return_sequences= True))
regressor_rnn.add(Dropout(rate = 0.2))

# third GRU layer with Dropout-Regularization
regressor_rnn.add(GRU(units= 50, return_sequences= True))
regressor_rnn.add(Dropout(rate = 0.2))

# forth GRU layer (last GRU layer) with Dropout-Regularization
regressor_rnn.add(GRU(units= 50))
regressor_rnn.add(Dropout(rate = 0.2))

# the output Layer
regressor_rnn.add(Dense(units = 1))


# ------ Compiling the Model ----------
regressor_rnn.compile(optimizer='rmsprop', loss='mean_squared_error')


# ------- Fit the model to Training dataset (model learns) ----------
# The training will happen in this part.
regressor_rnn.fit(X_train, y_train, epochs= 100, batch_size=32)



# ------------------- Making Prediction ---------------------
# getting the Real Google stock price of 2017-January from Google_Stock_Price_Test.csv
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv') 
real_stock_price = dataset_test.iloc[:, 1:2].values

# getting the Predicted Google stock price of 2017-January from our trained model
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs_for_predict = dataset_total[len(dataset_total)-len(dataset_test)-60 : ].values
# values makes it array, otherwise it is just a series
# it is 60 + 20 = 80 stock-prices, combining october, november, December 2016 and january 2017
print(len(inputs_for_predict)) 

# simple reshape for the input: to make it a vector, similar job as 'iloc'
inputs_for_prd = inputs_for_predict.reshape(-1, 1)
inputs_for_prd = sc.transform(inputs_for_prd) # scaling, only "transform" , no "fit"


# creating the data structure for test-set
test_data_size = inputs_for_prd.shape[0]
X_test = []
# There is no "y_test = []" we'll predict it

for i in range(60, test_data_size):
    X_test.append(inputs_for_prd[(i-60):i, 0])

# Convert X_test to NumPy array
X_test = np.array(X_test)

# Reshaping 
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# ---- Making Prediction !!! -----
y_pred = regressor_rnn.predict(X_test)
predicted_stock_price = sc.inverse_transform(y_pred)


# ------------------- Visualizing the Result ---------------------
print(f"\n------ Real Stock prices--------- \n ")
print(f"{real_stock_price}")
print(f"\n------ --------- \n ")
print(f"\n------ Predicted Stock prices--------- \n ")
print(f"{predicted_stock_price}")
print(f"\n------ --------- \n ")


plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock price of January 2017')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock price of January 2017')
plt.title('Google Stock price prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock price')
plt.legend()
plt.show()



# python prctc_GRU_rnn.py