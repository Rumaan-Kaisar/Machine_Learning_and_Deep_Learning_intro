# Reinforcement lrarning : ----------  Thompson Sampling. ---------  
#  Ad Click through rate optimization

import pandas as pd
import numba as np
import matplotlib.pyplot as plt


# importing dataset
dataSet = pd.read_csv("Ads_CTR_Optimisation.csv")


# Thompson Sampling Implemnetation. 
"""    # 2 parameters are important: 
        # no. of bandit d, 
        # No. of trials N, 
        """

import random
# here d is No. of Ads or Number of Bandits
d = 10
N = 10000   # Total number of trials

"""
numbers_of_selections = [0]*d
numbers_of_rewards = [0]*d

changed to 

        # no. of reward of Ad i at trial n i.e. reward = 1, 
        # No. of punishment of Ad i at trial n i.e. reward = 0,
"""
reward_count_of_ads = [0]*d
punish_count_of_ads = [0]*d

slected_ads = []
total_reward = 0


for n in range (0, N):
    slct_ad = 0
    max_random_beta = 0

    for i in range (0, d):
        random_beta = random.betavariate(reward_count_of_ads[i]+1, punish_count_of_ads[i]+1)

        if  random_beta > max_random_beta:
            max_random_beta = random_beta
            slct_ad = i
    
    # storing selected Ad
    slected_ads.append(slct_ad)

    # updating "reward_count_of_ads" and "punish_count_of_ads" of selected Ad "slct_ad"
    reward = dataSet.values[n, slct_ad] # generating reward-simulation from given Dataset
    if reward == 1:
        reward_count_of_ads[slct_ad] += 1
    else:
        punish_count_of_ads[slct_ad] += 1
    
    total_reward += reward


#visualizing the result
plt.hist(slected_ads, rwidth=0.85)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()


# python prctc_tmpsn_sml.py