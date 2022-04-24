# Reinforcement lrarning : ----------  UCB. ---------  Select an add by Click on Ad 

import pandas as pd
import numba as np
import matplotlib.pyplot as plt


# importing dataset
dataSet = pd.read_csv("Ads_CTR_Optimisation.csv")


# UCB Implemnetation. 
    # 3 parameters are important: 
        # no. of bandit d, 
        # No. of trials N, 
        # initial max_upper_bound (1e400)

import math
# here d is No. of Ads or Number of Bandits
d = 10
N = 10000   # Total number of trials

numbers_of_selections = [0]*d
numbers_of_rewards = [0]*d
slected_ads = []
total_reward = 0


for n in range (0, N):
    ad_max_ucb = 0
    max_upper_bound = 0

    for i in range (0, d):
        if(numbers_of_selections[i] > 0):
            avg_reward = numbers_of_rewards[i]/numbers_of_selections[i]
            # log(n+1) because of index
            delta = math.sqrt((3*math.log(n+1))/(2*numbers_of_selections[i]))
            upper_conf_bound =  avg_reward + delta
        else:
            upper_conf_bound = 1e400 # i.e. 10^400

        if upper_conf_bound > max_upper_bound:
            max_upper_bound = upper_conf_bound
            ad_max_ucb = i
    
    # storing selected Ad
    slected_ads.append(ad_max_ucb)

    # updating "numbers_of_selections" and "numbers_of_rewards" of selected Ad "ad_max_ucb"
    numbers_of_selections[ad_max_ucb] += 1
    reward = dataSet.values[n, ad_max_ucb] # generating reward-simulation from given Dataset
    numbers_of_rewards[ad_max_ucb] += reward

    total_reward += reward


#visualizing the result
plt.hist(slected_ads, rwidth=0.85)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()


# python prctc_UCB.py