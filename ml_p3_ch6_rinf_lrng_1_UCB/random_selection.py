# Random Selection

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Random Selection
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    # in this case "reward" indicates the total clicks if we select the ads "randomly".
    # in UCB we do not select ads randomly.
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected, rwidth=0.85)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

"""
The parameter rwidth specifies the width of your bar relative to the width of your bin. For example, if your bin width is say 1 and rwidth=0.5, the bar width will be 0.5. On both side of the bar you will have a space of 0.25.
"""