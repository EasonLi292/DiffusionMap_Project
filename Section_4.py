'''
For section 4, wasn't able to get the exact same results, I think maybe my epsilon could've
been better chosen, however, it does show the level sets by subtracting the gradient dot product
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs

# Define polar vectors
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2 * np.pi, 100)
phi_grid, theta_grid = np.meshgrid(phi, theta)

# f = sin(12theta)
f = np.sin(12 * theta_grid)

# Convert spherical to Cartesian coordinates
x = np.sin(theta_grid) * np.cos(phi_grid)
y = np.sin(theta_grid) * np.sin(phi_grid)
z = np.cos(theta_grid)

# Plot the function on the sphere
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(
    x, y, z,
    facecolors=plt.cm.jet((f - f.min()) / (f.max() - f.min())),
    rcount=100, ccount=100
)
ax.set_title("f = sin(12θ) on Sphere")
plt.show()

# Flatten the coordinates
coords = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T  # Shape (N, 3)

# Compute the derivative of f with respect to theta
df_dtheta = 12 * np.cos(12 * theta_grid)  # Shape (100, 100)

# Compute cos and sin of theta and phi
cos_theta = np.cos(theta_grid)
sin_theta = np.sin(theta_grid)
cos_phi = np.cos(phi_grid)
sin_phi = np.sin(phi_grid)

# Compute the components of the theta unit vector in Cartesian coordinates
e_theta_x = cos_theta * cos_phi
e_theta_y = cos_theta * sin_phi
e_theta_z = -sin_theta

# Compute the gradient components in Cartesian coordinates
grad_f_x = df_dtheta * e_theta_x
grad_f_y = df_dtheta * e_theta_y
grad_f_z = df_dtheta * e_theta_z

# Flatten the gradient components
grad_f_x_flat = grad_f_x.ravel()
grad_f_y_flat = grad_f_y.ravel()
grad_f_z_flat = grad_f_z.ravel()

# Stack gradient vectors into a single array
grad_f = np.vstack([grad_f_x_flat, grad_f_y_flat, grad_f_z_flat]).T  # Shape (N, 3)

# Compute differences between all pairs of points using np.subtract
coords_i = coords[:, np.newaxis, :]  # Shape (N, 1, 3)
coords_j = coords[np.newaxis, :, :]  # Shape (1, N, 3)
diffs = np.subtract(coords_i, coords_j)  # Shape (N, N, 3)

# Compute squared distances
dists_sq = np.sum(diffs ** 2, axis=2)  # Shape (N, N)

# Compute distances
dists = np.sqrt(dists_sq)  # Shape (N, N)

# Exclude zero distances and compute median
nonzero_dists = dists[dists > 0]
epsilon = np.median(nonzero_dists) ** 2

# Compute dot products grad_f(x_i) ⋅ (x_i - x_j)
dot_products = np.einsum('ik,ijk->ij', grad_f, diffs)  # Shape (N, N)

# Compute the kernel with the new epsilon
regular_diffusion = -dists_sq / epsilon
directional_diffusion = (dot_products ** 2) / epsilon 
kernel = np.exp(regular_diffusion - directional_diffusion)

# Normalize the kernel to create a Markov matrix
row_sums = kernel.sum(axis=1)
P = kernel / row_sums[:, np.newaxis]

# Ensure real eigenvalues and eigenvectors
vals, vecs = eigs(P.T, k=10, which='LR')
vals = vals.real
vecs = vecs.real

# Sort eigenvalues and eigenvectors
idx = vals.argsort()[::-1]
vals = vals[idx]
vecs = vecs[:, idx]

# Eigenfunction corresponding to the second largest eigenvalue (first nontrivial)
psi1 = vecs[:, 1]
psi1_reshaped = psi1.reshape(theta_grid.shape)

# Plot the first nontrivial eigenfunction on the sphere
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(
    x, y, z,
    facecolors=plt.cm.jet((psi1_reshaped - psi1_reshaped.min()) / (psi1_reshaped.max() - psi1_reshaped.min())),
    rcount=100, ccount=100
)
ax.set_title("First Nontrivial Eigenfunction on Sphere")
plt.show()
