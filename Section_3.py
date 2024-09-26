import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse.linalg import eigs
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# Number of points
N = 1000

# Torus parameters
R = 3  # Major radius
r = 1  # Minor radius

# Parameters controlling the spiral
k = 7  # Number of twists around the tube
l = 1  # Number of times the curve winds around the torus

t = np.linspace(0, 2 * np.pi, N)

# Define the slowly varying density function over half a period
density_function = 0.55 + 0.45 * np.cos(t)  # Varies slowly over half a period
density_function /= density_function.sum()  # Normalize

# Generate cumulative density function
cumulative_density = np.cumsum(density_function)
cumulative_density /= cumulative_density[-1]

# Generate non-uniformly sampled t values
random_values = np.sort(np.random.rand(N))
t_nonuniform = np.interp(random_values, cumulative_density, t)

# Define the toroidal spiral curve
x = (R + r * np.cos(k * t_nonuniform)) * np.cos(l * t_nonuniform)
y = (R + r * np.cos(k * t_nonuniform)) * np.sin(l * t_nonuniform)
z = r * np.sin(k * t_nonuniform)

# Stack the points into an array
points = np.vstack((x, y, z)).T

# Normalize t_nonuniform for color mapping
t_normalized = (t_nonuniform - t_nonuniform.min()) / (t_nonuniform.max() - t_nonuniform.min())
cmap = cm.get_cmap('hsv')
colors = cmap(t_normalized)

# Compute pairwise distances
distances = euclidean_distances(points, points)

#use median for epsilon
epsilon = np.median(distances) ** 2
K = np.exp(-distances ** 2 / epsilon)

# Degree function
d = K.sum(axis=1)

'''
i think this part might be wrong
The outer product ensures that each pair of points 
is weighted based on their local densities (via their degrees). 
'''

# Anisotropic Kernels
K_alpha_0 = K / np.outer(d ** 0, d ** 0)
K_alpha_1 = K / np.outer(d ** 1, d ** 1)

# Normalize to obtain transition probabilities
Z_alpha_0 = K_alpha_0.sum(axis=1)
P_alpha_0 = K_alpha_0 / Z_alpha_0[:, np.newaxis]

Z_alpha_1 = K_alpha_1.sum(axis=1)
P_alpha_1 = K_alpha_1 / Z_alpha_1[:, np.newaxis]

# Compute eigenvectors
num_eigenvectors = 10

# For alpha = 0
vals_0, vecs_0 = eigs(P_alpha_0, k=num_eigenvectors, which='LM')

# For alpha = 1
vals_1, vecs_1 = eigs(P_alpha_1, k=num_eigenvectors, which='LM')

# Visualization
# Original 3D Curve 
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=20)
ax.set_title('Original 3D Toroidal Spiral Curve with Color-Coded Points')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# Density of Points Along the Curve
plt.figure(figsize=(12, 6))
plt.scatter(t_nonuniform, d, c=colors, s=20)
plt.title('Density of Points Along the Curve')
plt.xlabel('Curve Parameter (t)')
plt.ylabel('Density (d)')
plt.show()

# Embedding via Graph Laplacian (α = 0)
embedding_alpha_0 = vecs_0[:, 1:3]  # Use the first two non-trivial eigenvectors

plt.figure(figsize=(8, 6))
plt.scatter(embedding_alpha_0[:, 0], embedding_alpha_0[:, 1], c=colors, s=20)
plt.title('Embedding via Graph Laplacian')
plt.xlabel('Eigenvector 1')
plt.ylabel('Eigenvector 2')
plt.axis('equal')
plt.show()

# Embedding via Laplace–Beltrami Approximation (α = 1)
embedding_alpha_1 = vecs_1[:, 1: 3]  # Use the first two non-trivial eigenvectors

plt.figure(figsize=(8, 6))
plt.scatter(embedding_alpha_1[:, 0], embedding_alpha_1[:, 1], c=colors, s=20)
plt.title('Embedding via Laplace–Beltrami Approximation')
plt.xlabel('Eigenvector 1')
plt.ylabel('Eigenvector 2')
plt.axis('equal')
plt.show()

from pydiffmap import diffusion_map as dm
# Compute pairwise distances to estimate epsilon
distances = euclidean_distances(points, points)
eps = np.median(distances) ** 2

# Instantiate the Diffusion Map object
dmap_alpha_0 = dm.DiffusionMap.from_sklearn(alpha=0, epsilon=eps, n_evecs=10)
dmap_alpha_1 = dm.DiffusionMap.from_sklearn(alpha=1, epsilon=eps, n_evecs=10)

# Fit the model to the data
dmap_alpha_0.fit(points)
dmap_alpha_1.fit(points)

# Extract the diffusion map embedding (skip the first eigenvector)
# Take the real part since eigenvectors may be complex
embedding_alpha_0 = np.real(dmap_alpha_0.evecs[:, 1:3])
embedding_alpha_1 = np.real(dmap_alpha_1.evecs[:, 1:3])

# Visualization
# Original 3D Curve
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=20)
ax.set_title('Original 3D Toroidal Spiral Curve with Color-Coded Points')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# Embedding via Graph Laplacian (α = 0)
plt.figure(figsize=(8, 6))
plt.scatter(embedding_alpha_0[:, 0], embedding_alpha_0[:, 1], c=colors, s=20)
plt.title('Embedding via Graph Laplacian (α = 0)')
plt.xlabel('Eigenvector 1')
plt.ylabel('Eigenvector 2')
plt.axis('equal')
plt.show()

# Embedding via Laplace–Beltrami Approximation (α = 1)
plt.figure(figsize=(8, 6))
plt.scatter(embedding_alpha_1[:, 0], embedding_alpha_1[:, 1], c=colors, s=20)
plt.title('Embedding via Laplace–Beltrami Approximation (α = 1)')
plt.xlabel('Eigenvector 1')
plt.ylabel('Eigenvector 2')
plt.axis('equal')
plt.show()