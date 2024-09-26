#Diffusion map
#!/usr/bin/env python3 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import fractional_matrix_power


'''
SECTION 2
'''
#Make_blobs: generate data with 900 samples, 3 centers in 2D space, 

centers = [(-1.5, 0), (1, -1), (1, 1)]
X,y = make_blobs(n_samples = 900,n_features = 2,centers = centers, cluster_std=0.4, random_state=1)

# Construct the graph with Gaussian weights, epsilon = 0.7
epsilon = 0.7
pairwise_dists = squareform(pdist(X, 'euclidean'))
Gaussian_weights = np.exp(-pairwise_dists**2 / epsilon)

#markov matrix
#each row sum to 1 to represent probability in Markov Chain
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

#Figure 2
'''
Since I don't have the collection of images, I manually create the images, with different rotations
'''
from PIL import Image, ImageDraw, ImageFont
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse.linalg import eigs

def generate_text_images(num_angles=100, text='3D', image_size=(100, 100), font_size=72):
    images = [] #store generated images of 100 angles
    angles = np.linspace(0, 360, num_angles, endpoint=False)
    
    # Create a base image with the text
    base_image = Image.new('L', image_size, color='white')  # 'L' mode for grayscale
    draw = ImageDraw.Draw(base_image)
    font = ImageFont.load_default()
    
    # Get text size and draw it in the center
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    text_x = (image_size[0] - text_width) / 2
    text_y = (image_size[1] - text_height) / 2
    draw.text((text_x, text_y), text, fill=0, font=font)
    
    for angle in angles:
        # Rotate the image and add flattened version to images
        rotated_image = base_image.rotate(angle, resample=Image.BILINEAR)
        images.append(np.array(rotated_image).flatten())
    
    return np.array(images), angles


images, angles = generate_text_images()

scaler = StandardScaler()
images_normalized = scaler.fit_transform(images)

#Construct the affinity matrix
pairwise_dists = pairwise_distances(images_normalized, metric='euclidean')
#ideally choose a sigma value that's in the median so the affinity is balanced between close and far distances
#important
sigma = np.median(pairwise_dists)
affinity_matrix = np.exp(-pairwise_dists ** 2 / (2 * sigma ** 2))

# Build the transition matrix
degree_matrix = np.sum(affinity_matrix, axis=1)
P = affinity_matrix / degree_matrix[:, None]

# Compute eigenvalues and eigenvectors
num_eigenvectors = 5
eigenvalues, eigenvectors = eigs(P.T, k=num_eigenvectors, which='LR')
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Diffusion coordinates
psi = eigenvectors[:, 1:].real
lambda_vals = eigenvalues[1:].real
t = 1
diffusion_coords = psi * (lambda_vals ** t)

# Visualize
plt.figure(figsize=(8, 6))
scatter = plt.scatter(diffusion_coords[:, 0], diffusion_coords[:, 1], c=angles, cmap='hsv')
plt.colorbar(scatter, label='Rotation Angle (degrees)')
plt.xlabel(r'$\psi_1$')
plt.ylabel(r'$\psi_2$')
plt.title('Diffusion Map Embedding of Rotated "3D" Text Images')
plt.show()

'''
Since the images are manually created, I did not know how to generate images with topology differences, thus the result looks different from the lecture. 
Thus the resulting graph looks different as only angle relatedness are shown, however, the circular structure of the rotating angles can be visualized clearly, 
event though some features may be different
'''
