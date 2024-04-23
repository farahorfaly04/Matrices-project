from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from utils import create_matrices, create_vectors, display_matrices, calculate_mean_face, display_mean_face, calculate_covariance_matrix, calculate_eigen, calculate_weights, euclidean_distance, calculate_distances

# List of matrices for training images
train_matrices = create_matrices('Faces')
test_matrices = create_matrices('Faces')

# Display an example matrix
#display_matrix(train_matrices_array[0])

# Display all matrices as images
display_matrices(train_matrices)
display_matrices(test_matrices)

# Convert images to vectors
train_vectors = create_vectors(train_matrices)
test_vectors = create_vectors(test_matrices)

# Calculate mean face image
train_vectors_matrix, mean_face_vector = calculate_mean_face(train_vectors)

display_mean_face(mean_face_vector)

# Subtract mean_face_vector from train_vectors
train_vectors_minus_mean = [(vector - mean_face_vector, filename) for vector, filename in train_vectors]

# Subtract mean_face_vector from test_vectors
test_vectors_minus_mean = [(vector - mean_face_vector, filename) for vector, filename in test_vectors]

# Get the covariance Matrix 
covariance_matrix = calculate_covariance_matrix(train_vectors_matrix)

# Calculate eigen
eigenvalues, eigenvectors, eigenfaces = calculate_eigen(covariance_matrix)

train_weights = calculate_weights(eigenfaces, train_vectors_minus_mean)


"""# Testing"""

test_weights = calculate_weights(eigenfaces, test_vectors_minus_mean)

# Calculate euclidean distances 
all_distances = calculate_distances(test_weights, train_weights)
for distances in all_distances:
    for distance, train, test in distances:
        print(distance)
        print(train)
        print(test)
        print("\n\n\n")
    
# Find the maximum distance
max_distance = max(max(distance_tuple[0] for distance_tuple in distances) for distances in all_distances)
print(max_distance)
# Define the threshold (T) as half of the maximum distance
T = max_distance / 3

# Classify the testing images
for i, distances in enumerate(all_distances):
    distances_values = [distance[0] for distance in distances]
    min_distance = min(distances_values)
    if min_distance < T:
        min_index = np.argmin(distances_values)
        corresponding_train_name = distances[min_index][1]
        corresponding_test_name = distances[min_index][2]  

        # Load the images
        test_image = Image.open(f'Olivetti/test/{corresponding_test_name}')
        train_image = Image.open(f'Olivetti/train/{corresponding_train_name}')

        # Create a figure to display both images side by side
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(test_image, cmap='grey')
        plt.title(f"Test Image {i+1}")
        plt.axis('off')  # Hide axes

        plt.subplot(1, 2, 2)
        plt.imshow(train_image, cmap='grey')
        plt.title(f"Matched with {corresponding_train_name}")
        plt.axis('off')  # Hide axes

        plt.show()
    else:
        print(f"Testing image {i+1} is unknown")