'''
No idea how to do the helix equation in the article lol
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse.linalg import eigsh

# noisy 3D helix
N = 400  
t = np.linspace(0, 4 * np.pi, N) 
noise_level = 0.1  

# Helix parameters
x = np.cos(t) + noise_level * np.random.randn(N)
y = np.sin(t) + noise_level * np.random.randn(N)
z = t + noise_level * np.random.randn(N)

# Stack into an array
points = np.vstack((x, y, z)).T

# Color mapping based on t
t_normalized = (t - t.min()) / (t.max() - t.min())
cmap = plt.cm.get_cmap('hsv')
colors = cmap(t_normalized)

def diffusion_map(data, n_components=2, alpha=0.5):

    distances = pairwise_distances(data, metric='euclidean')

    # WE CAN CHANGE E TO SHOW THE RELATIONSHIP OF EPSILON AND PERTURBATION
    # VALID AS LONG AS SQRT(E) > PERTURBATION
    epsilon = np.median(distances)

    # Compute the kernel matrix
    # Note gaussian kernel is a smooth function, thus
    # we can linearize the effect of perturbations:
    K = np.exp(-distances ** 2 / (2 * epsilon ** 2))

    # Compute degree matrix
    d = np.sum(K, axis=1)

    # Normalize the kernel
    D_alpha = np.diag(d ** (-alpha))
    K_normalized = D_alpha @ K @ D_alpha

    # Compute the Markov matrix
    d_tilde = np.sum(K_normalized, axis=1)
    D_tilde_inv = np.diag(1 / d_tilde)
    P = D_tilde_inv @ K_normalized

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigsh(P, k=n_components + 1, which='LM')

    # Sort eigenvalues and eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Skip the first eigenvector
    embedding = eigenvectors[:, 1:n_components + 1] * eigenvalues[1:n_components + 1]

    return embedding.real

# Apply diffusion map
embedding = diffusion_map(points, n_components=2, alpha=1)

# Noisy
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=20)
ax.set_title('Noisy 3D Helix')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 2D embedding
ax2 = fig.add_subplot(122)
ax2.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=20)
ax2.set_title('2D Diffusion Map Embedding')
ax2.set_xlabel('Embedding Dimension 1')
ax2.set_ylabel('Embedding Dimension 2')
ax2.axis('equal')

plt.tight_layout()
plt.show()
