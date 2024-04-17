from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from utils import create_matrices, create_vectors, display_matrix, display_matrices, calculate_mean_face, display_mean_face, calculate_covariance_matrix, calculate_eigen, calculate_weights

# List of matrices for training images
train_matrices = create_matrices('Images/Train Data')
test_matrices = create_matrices('Images/Test Data')

# Convert train_matrices to a NumPy array
train_matrices_array = np.array(train_matrices)

# Display an example matrix
display_matrix(train_matrices_array[0])

# Display all matrices as images
display_matrices(test_matrices)

# Convert images to vectors
train_vectors = create_vectors(train_matrices)
test_vectors = create_vectors(test_matrices)

# Calculate mean face image
train_vectors_matrix, mean_face_vector = calculate_mean_face(train_vectors)

display_mean_face(mean_face_vector)

# Subtract mean_face_vector from train_vectors
train_vectors_minus_mean = [vector - mean_face_vector for vector in train_vectors]

# Subtract mean_face_vector from test_vectors
test_vectors_minus_mean = [vector - mean_face_vector for vector in test_vectors]

# Get the covariance Matrix 
covariance_matrix = calculate_covariance_matrix(train_vectors_matrix)

# Display the covariance matrix
print(f"Covariance Matrix (C): {covariance_matrix}")

print("Dimensions of Covariance Matrix (C):", covariance_matrix.shape)

eigenvalues, eigenvectors, eigenfaces = calculate_eigen(covariance_matrix)

# Display the subset of eigenvalues
print("Subset of Eigenvalues:")
print(eigenvalues)

# Display the subset of eigenvectors
print("\nSubset of Eigenvectors:")
print(eigenvectors)

# Display the normalized eigenvectors
print("Eigenfaces (Normalized Eigenvectors):")
print(eigenfaces)

eigenfaces_reshaped, weights_list = calculate_weights(eigenfaces, train_vectors_minus_mean)

# Print the weights for each image
for i, weights in enumerate(weights_list):
    print(f"Weights for image {i+1}:")
    print(weights)
    print()

"""# Testing"""

# List of testing images
testing_images = test_vectors_minus_mean

# List to store weights for each testing image
testing_weights_list = []

# Calculate weights for each testing image
for image_vector in testing_images:
    # Calculate the weights for the testing image
    testing_weights = np.dot(eigenfaces_reshaped, image_vector)

    # Append weights to the list
    testing_weights_list.append(testing_weights)

# Print the weights for each testing image
for i, testing_weights in enumerate(testing_weights_list):
    print(f"Weights for testing image {i+1}:")
    print(testing_weights)
    print()

# Calculate the pairwise distances between all training weights
pairwise_distances = []
for i in range(len(testing_weights_list)):
    distances = [np.linalg.norm(testing_weights_list[i] - weights) for weights in weights_list]
    pairwise_distances.append(distances)

# Find the maximum distance
max_distances = [max(distances) for distances in pairwise_distances]
max_distance = max(max_distances)

# Define the threshold (T) as half of the maximum distance
T = max_distance / 2

# Classify the testing images
for i, testing_weights in enumerate(testing_weights_list):
    min_distance = min(pairwise_distances[i])
    if min_distance < T:
        min_index = np.argmin(pairwise_distances[i])
        print(f"Testing image {i+1} belongs to class {min_index+1}")
    else:
        print(f"Testing image {i+1} is unknown")

