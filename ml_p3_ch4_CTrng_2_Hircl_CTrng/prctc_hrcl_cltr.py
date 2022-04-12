# Hierachical Clustering

# libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# importing data
dataSet = pd.read_csv("Mall_Customers.csv")
X = dataSet.iloc[:, [3, 4]].values
# There is no y for clustering


# Dendrogram :: Finding optimal noumber of clusters using "Dendrogram"
import scipy.cluster.hierarchy as schr
dendrgm = schr.dendrogram(schr.linkage(X, method="ward"))
plt.title("Dendrogram")
plt.xlabel("Number of Clusters")
plt.ylabel("Euclidean distance")
plt.show()


# creating cluster with optimal "n_clusters". From "DENDROGRAM" we figured out that 5 is the optimal number of clusters
from sklearn.cluster import AgglomerativeClustering
hrcl_cluster_genrt = AgglomerativeClustering(n_clusters = 5, affinity="euclidean", linkage='ward')
y_hrcl = hrcl_cluster_genrt.fit_predict(X)

# plotting the cluster 
plt.scatter(X[y_hrcl == 0, 0], X[y_hrcl == 0, 1], s = 100, c = "red", label="cluster 1")
plt.scatter(X[y_hrcl == 1, 0], X[y_hrcl == 1, 1], s = 100, c = "blue", label="cluster 2")
plt.scatter(X[y_hrcl == 2, 0], X[y_hrcl == 2, 1], s = 100, c = "green", label="cluster 3")
plt.scatter(X[y_hrcl == 3, 0], X[y_hrcl == 3, 1], s = 100, c = "cyan", label="cluster 4")
plt.scatter(X[y_hrcl == 4, 0], X[y_hrcl == 4, 1], s = 100, c = "pink", label="cluster 5")

# No centroids in Hierarchical clustering
# plt.scatter(hrcl_cluster_genrt.cluster_centers_[:, 0], hrcl_cluster_genrt.cluster_centers_[:, 1], s=300, c="black", label = "Centroids")
plt.title("Clusters Of Clients")
plt.xlabel("Annual income ($)")
plt.ylabel("Spending score (1-100)")
plt.legend()
plt.show()


# python prctc_hrcl_cltr.py