# ----------- RBM : Recommender ---------------

# Importing the libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# importing the dataset
movies = pd.read_csv("./movie_lens_1m/movies.dat", sep= "::", header=None, engine="python", encoding="latin-1")
useRs = pd.read_csv("./movie_lens_1m/users.dat", sep= "::", header=None, engine="python", encoding="latin-1")
RaTings = pd.read_csv("./movie_lens_1m/ratings.dat", sep= "::", header=None, engine="python", encoding="latin-1")

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



# Converting the ratings into binary ratings: 1 (liked),  0 (not-liked)
train_set_tensor[train_set_tensor == 0] = -1
        # torch doesn't support combined condition
train_set_tensor[train_set_tensor == 1] = 0   
train_set_tensor[train_set_tensor == 2] = 0
train_set_tensor[train_set_tensor >= 3] = 1

test_set_tensor[test_set_tensor == 0] = -1
test_set_tensor[test_set_tensor == 1] = 0   
test_set_tensor[test_set_tensor == 2] = 0
test_set_tensor[test_set_tensor >= 3] = 1

# train_tensor_to_view = train_set_tensor.detach().cpu().numpy()
# test_tensor_to_view = test_set_tensor.detach().cpu().numpy()
train_tensor_to_view = train_set_tensor.numpy()
test_tensor_to_view = test_set_tensor.numpy()


# -------- Creating the architecture of the Neural Netwark --------
class RBM():
    def __init__(self, nv, nh) :
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)

    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)



# creating RBM model
nv = len(train_set_tensor[0])
nh = 100
batch_size = 100

rbm = RBM(nv, nh)


# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.0
    for u_id in range(0, nb_users - batch_size, batch_size):
        vk = train_set_tensor[u_id:(u_id + batch_size)]     # input vector
        v0 = train_set_tensor[u_id:(u_id + batch_size)]     # target vector

        ph0,_ = rbm.sample_h(v0)    # initial probabilities

            # applying k-step contrastive divergence
        for k in range(10):
            _,hk = rbm.sample_h(vk)     # sampling hidden nodes
            _,vk = rbm.sample_v(hk)     # sampling visible nodes
            vk[v0 < 0] = v0[v0 < 0]     # preventing updates to unrated nodes

        phk,_ = rbm.sample_h(vk)    # probabilities after k-step 
        rbm.train(v0, vk, ph0, phk) # train RBM model

        # Error rate
        train_loss += torch.mean(torch.abs(vk[v0 >= 0] - v0[v0 >= 0]))
        s += 1.0
    
    loSS = train_loss/s
    print(f"Epoch no. {epoch}.\t loss = {loSS}")



# Evaluating the RBM on Test-set
test_set_ratings_list = []
pridicted_rating_list = []

test_loss = 0 
cnt = 0.0
for u_id in range(nb_users):
    v = train_set_tensor[u_id:(u_id + 1)]     # input vector from training-set
    vt = test_set_tensor[u_id:(u_id + 1)]     # target vector from test-set

    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)     # sampling hidden nodes
        _,v = rbm.sample_v(h)     # sampling visible nodes

        # Error rate
        test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
        cnt += 1.0

    # creating list of original & predicted ratings
    original_test_set_ratings = vt.numpy()
    test_set_ratings_list.append(original_test_set_ratings)
    predicted_ratings = v.numpy()
    pridicted_rating_list.append(predicted_ratings)

eval_loSS = test_loss/cnt
print(f"Evaluation or Test loss = {eval_loSS}")


# python prctc_RBM.py