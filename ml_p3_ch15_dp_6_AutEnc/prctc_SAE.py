# ----------- AE : Recommender. SAE Stacked-Auto-Encoder ---------------

# Importing the libraries
from turtle import clone
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


# ---------- importing the dataset -----------

# preparing the training set and test set
training_set = pd.read_csv("./movie_lens_100k/u1.base", delimiter="\t")
train_set = np.array(training_set, dtype="int")

ts_set = pd.read_csv("./movie_lens_100k/u1.test", delimiter="\t")
test_set = np.array(ts_set, dtype="int")


# Getting the number of Users and Movies
nb_users = int(max(max(train_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(train_set[:, 1]), max(test_set[:, 1])))


# converting the data into an array with users in lines and movies in column.
def conVert(data):
    new_data = []
    for id_user in range(1, nb_users + 1):
        # use "data[:, 0] == id_user" as condition over movie column "data[:, 1]"
        id_movies = data[:, 1][data[:, 0]== id_user]    # returns a list

        # use "data[:, 0] == id_user" as condition over ratins column "data[:, 2]"
        id_ratings = data[:, 2][data[:, 0]== id_user] 

        # vector of zeros
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        
        new_data.append(list(ratings))
    
    return new_data

trn_set_cnvt = conVert(train_set)
tst_set_cnvt = conVert(test_set)


# Converting the data into Torch Tensosrs. Following are the Tensors of ratings
train_set_tensor = torch.FloatTensor(trn_set_cnvt)
test_set_tensor = torch.FloatTensor(tst_set_cnvt)




# -------- Creating the architecture of the Neural Netwark for SAE --------
class StackedAutoEncoders(nn.Module):
    def __init__(self, ):
        super(StackedAutoEncoders, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

sae = StackedAutoEncoders()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay= 0.5)


# ---- Training the SAE model ----
nb_epoch = 200
for epoch in range(1, nb_epoch+1):
    train_loss = 0
    s = 0.0
    for id_user in range(nb_users):
        input = Variable(train_set_tensor[id_user]).unsqueeze(0)
        target = input.clone()
        
        non_zero_ratings = torch.sum(target.data > 0)
        if non_zero_ratings > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(non_zero_ratings + 1e-10)
            
            loss.backward()
            train_loss += np.sqrt((loss.data)*mean_corrector)
            s += 1.0
            optimizer.step()

    train_loss_normalized = train_loss/s
    print(f"Epoch : {epoch}, Loss = {train_loss_normalized}")



# ---- Testing the SAE model on Test-set ----
# Evaluating the SAE on Test-set
test_set_ratings_list = []
pridicted_rating_list = []

test_loss = 0 
cnt = 0.0
for u_id in range(nb_users):
    v = Variable(train_set_tensor[u_id]).unsqueeze(0)     # input vector from training-set
    vt = Variable(test_set_tensor[u_id]).unsqueeze(0)     # target vector from "test-set"

    non_zero_ratings_tst = torch.sum(vt.data > 0)
    if non_zero_ratings_tst > 0:
        tst_output = sae(v)
        vt.require_grad = False

        tst_output[vt == 0] = 0     # avoiding unrated movies in test-set
        loss_tst = criterion(tst_output, vt)    # comparing

        mean_corrector_tst = nb_movies/float(non_zero_ratings_tst + 1e-10)
        test_loss += np.sqrt((loss_tst.data)*mean_corrector_tst)
        cnt += 1.0

    # creating list of original & predicted ratings
        """  we have to use detach() to convert tensor to Numpy-array
                 because our tensor requires "grad" now. 
             We did it in RBM already but without detach() """
             
    original_test_set_ratings = vt.detach().numpy() 
    test_set_ratings_list.append(original_test_set_ratings)
    predicted_ratings = tst_output.detach().numpy()
    pridicted_rating_list.append(predicted_ratings)

eval_loSS = test_loss/cnt
print(f"Evaluation or Test loss = {eval_loSS}")



# python prctc_SAE.py