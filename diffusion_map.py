#Diffusion map 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import fractional_matrix_power


'''
basic k-means clustering
'''

#Make_blobs: generate data with 900 samples, 3 centers in 2D space, 
#copying the article for centers

centers = [(-1.5, 0), (1, -1), (1, 1)]
X,y = make_blobs(n_samples = 900,n_features = 2,centers = centers, cluster_std=0.4, random_state=1)

# Construct the graph with Gaussian weights
epsilon = 0.7
pairwise_dists = squareform(pdist(X, 'euclidean'))
Gaussian_weights = np.exp(-pairwise_dists**2 / epsilon)

#markov matrix
#each row sum to 1 to represent probability
row_sums = Gaussian_weights.sum(axis=1)
P = Gaussian_weights / row_sums[:, np.newaxis]

#t = 8, t = 64, t = 1024
P8 = np.linalg.matrix_power(P, 8)
P64 = np.linalg.matrix_power(P, 64)
P1024 = np.linalg.matrix_power(P, 1024)

#print
# Step 5: Visualize the results with help of ChatGPT
fixed_point = 0
def plot_diffusion_and_matrix(P_power, title, fixed_point):
    plt.figure(figsize=(12, 5))
    
    # Left plot: Diffusion intensity
    plt.subplot(1, 2, 1)
    diffusion_intensity = P_power[fixed_point, :]
    plt.scatter(X[:, 0], X[:, 1], c=diffusion_intensity, cmap='viridis')
    plt.colorbar()
    plt.title(f"Diffusion Intensity at {title}")
    
    # Right plot: Transition matrix
    plt.subplot(1, 2, 2)
    plt.imshow(P_power, cmap='viridis')
    plt.colorbar()
    plt.title(f"Transition Matrix {title}")
    
    plt.tight_layout()
    plt.show()

plot_diffusion_and_matrix(P8, "t = 8", fixed_point)
plot_diffusion_and_matrix(P64, "t = 64", fixed_point)
plot_diffusion_and_matrix(P1024, "t = 1024", fixed_point)

