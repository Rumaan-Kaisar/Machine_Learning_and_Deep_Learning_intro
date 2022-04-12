# WCSS : Within-Cluster Sum of Square

# K-Means Clustering

# libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# importing data
dataSet = pd.read_csv("Mall_Customers.csv")
X = dataSet.iloc[:, [3, 4]].values
# There is no y for clustering


# Finding optimal noumber of clusters using elbow method : WCSS
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    # setting paramter for cluster generator
    cluster_generator_elbow = KMeans(n_clusters = i, init="k-means++", random_state=0, max_iter=300, n_init=10) 
    cluster_generator_elbow.fit(X) # fit the independent data
    wcss.append(cluster_generator_elbow.inertia_) # capturing wccs data for each i
    # inertia_ :  Sum of squared distances of samples to their closest cluster center. Is actually "wcss"

# visualizing the elbow digram ith clusters vs wcss
# range(1, 11), wcss: both are lists
plt.plot(range(1, 11), wcss)
plt.title("The elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# creating cluster with optimal "n_clusters". From elbow-plot we figured out that 5 is the optimal number of clusters
k_mean_cluster_genrt = KMeans(n_clusters = 5, init="k-means++", random_state=0, max_iter=300, n_init=10)
y_k_mean_cluster = k_mean_cluster_genrt.fit_predict(X)

# plotting the cluster 
plt.scatter(X[y_k_mean_cluster == 0, 0], X[y_k_mean_cluster == 0, 1], s = 50, c = "red", label="cluster 1")
plt.scatter(X[y_k_mean_cluster == 1, 0], X[y_k_mean_cluster == 1, 1], s = 50, c = "blue", label="cluster 2")
plt.scatter(X[y_k_mean_cluster == 2, 0], X[y_k_mean_cluster == 2, 1], s = 50, c = "green", label="cluster 3")
plt.scatter(X[y_k_mean_cluster == 3, 0], X[y_k_mean_cluster == 3, 1], s = 50, c = "cyan", label="cluster 4")
plt.scatter(X[y_k_mean_cluster == 4, 0], X[y_k_mean_cluster == 4, 1], s = 50, c = "pink", label="cluster 5")

# centroids
plt.scatter(k_mean_cluster_genrt.cluster_centers_[:, 0], k_mean_cluster_genrt.cluster_centers_[:, 1], s=100, c="black", label = "Centroids")
plt.title("Clusters Of Clients")
plt.xlabel("Annual income ($)")
plt.ylabel("Spending score (1-100)")
plt.legend()
plt.show()


# python prctc_k_mns.py