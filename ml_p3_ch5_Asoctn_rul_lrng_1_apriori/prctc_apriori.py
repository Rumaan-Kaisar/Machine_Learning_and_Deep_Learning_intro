# -------- Association rule :: Apriori --------------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# importing data
dataset = pd.read_csv("Market_Basket_Optimisation.csv", header = None)

# creating "List of Lists": List of product-set
# there are 7500 rows of transections and 20 columns of product
transacTions = []
for i in range(0, 7501):
    # notice :: "List comprehension is used". Items are converted to strings
    # transacTions.append([str(dataset.values[i, j]) for j in range(0, 20)])
    transacTions.append([str(dataset.values[i, j]) for j in range(0, 20) if (str(dataset.values[i, j]) != "nan")])


# Train the apriori model
from apyori import apriori  
# rUles= apriori(transactions= transacTions, min_support=0.003, min_confidence = 0.2, min_lift=3, min_length=2, max_length=2)
rUles= apriori(transactions = transacTions, min_support=0.003, min_lift = 3, min_confidence=0.2, min_length = 2)
results= list(rUles)
for asocitn_item in results:
    print(f"{asocitn_item}\n")

# visualizing the rules
for item in results:  
    pair = item[0]   
    items = [x for x in pair]
    rul = "Rule: "
    for itm in items:
        rul += (itm + " -> " )
        
    print(rul)
    print(f"Rule: {items}")
    print("Support: " + str(item[1]))  
    print("Confidence: " + str(item[2][0][2]))  
    print("Lift: " + str(item[2][0][3]))  
    print("=====================================")  



# python prctc_apriori.py