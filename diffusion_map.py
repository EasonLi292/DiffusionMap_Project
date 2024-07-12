#Diffusion map 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

'''
basic k-means clustering, for seeing how it relate to jumps
'''

#Make_blobs: generate data with 500 samples, 3 centers in 2D space
X,y = make_blobs(n_samples = 500,n_features = 2,centers = 3,random_state=292)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=292)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

#print
fig = plt.figure(0)
plt.grid(True)
plt.scatter(X[:,0],X[:,1], c=y_kmeans, cmap='viridis')
#plt.show()

# Plot the cluster centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.show()