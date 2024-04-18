from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math

# Function to load and resize image
# Function to convert image to matrix
def load_image(image_path, size=(112,112)):
    img = Image.open(image_path).convert('L')  # Convert image to grayscale
    img = img.resize(size)
    matrix = np.array(img)
    return matrix

def create_matrices(folder_path):
    """Converts images into matrices"""
    matrices = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            image_path = os.path.join(dirpath, filename)
            matrix = load_image(image_path)    
            matrices.append((matrix, filename))

    return matrices

def create_vectors(matrices):
    """Convert the matrices into vectors """
    vectors = []
    for matrix, filename in matrices:
        vector = matrix.flatten().reshape(-1, 1)
        vectors.append((vector, filename))

    return vectors

def display_matrices(matrices):
    """Display matrices as images, with filenames."""
    num_images = len(matrices)
    num_cols = 10  # Number of columns in the display grid
    num_rows = (num_images + num_cols - 1) // num_cols  # Calculate number of rows based on number of images and columns

    plt.figure(figsize=(15, 3*num_rows))  # Adjust figsize based on number of rows

    for i, (matrix, filename) in enumerate(matrices):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(matrix, cmap='gray')
        plt.title(f'Image {filename}')
        plt.axis('off')

    plt.show()

def calculate_mean_face(vectors):
    """ Calculate the mean face image"""
    # Concatenate all vectors into a single matrix
    vectors_matrix = np.concatenate([v for v, _ in vectors], axis=1)

    # Calculate the mean face image
    mean_face_vector = np.mean(vectors_matrix, axis=1)
    mean_face_vector = mean_face_vector.flatten().reshape(-1,1)
    
    return vectors_matrix, mean_face_vector

def display_mean_face(mean_face_vector):
    """Display the mean face image"""
    
    # Reshape the mean face vector to its original shape (112x112)
    mean_face_image = mean_face_vector.reshape(112, 112)

    plt.imshow(mean_face_image, cmap='gray')
    plt.title('Mean Face Image')
    plt.axis('off')
    plt.show()

def calculate_covariance_matrix(vectors_matrix):
    """ Getting the covariance matrix"""
   
    # Combine all TEST vectors into one matrix A
    A = vectors_matrix
  
    # Calculate the covariance matrix C
    M = A.shape[1]  # Number of columns in A (number of vectors)
    C = (1/M) * np.dot(A, A.T)

    return C

def calculate_eigen(covariance_matrix):
    """ Finding the Eigenvalues and Eigenvectors of the covariance matrix"""

    # Specify the number of principal components (eigenvectors) to compute
    n_components = 40

    # Perform PCA and extract the subset of eigenvalues and eigenvectors
    pca = PCA(n_components=n_components)
    pca.fit(covariance_matrix)

    # Subset of eigenvalues
    eigenvalues = pca.explained_variance_

    # Subset of eigenvectors
    eigenvectors = pca.components_
    
    # Normalize the eigenvectors
    eigenfaces = [eigenvector / np.linalg.norm(eigenvector) for eigenvector in eigenvectors]

    return eigenvalues, eigenvectors, eigenfaces

def calculate_weights(eigenfaces, vectors_minus_mean):
    """ Calculating the weights of the train images"""

    # Reshape each eigenface into a row vector
    eigenfaces_reshaped = [(eigenface.flatten().reshape(1,-1)) for eigenface in eigenfaces]

    # Initialize list to hold all weights lists
    weights = []

    # Calculate weights for each image
    for vector, filename in vectors_minus_mean:
        # Calculate the weight for each eigenface and store it
        image_weights = [float(np.dot(eigenface, vector)) for eigenface in eigenfaces_reshaped]
        
        weights.append((image_weights, filename))

    return weights

def euclidean_distance(v1, v2):
    """Calculate the Euclidean distance between two lists of numbers."""
    distance = 0.0
    for i in range(len(v1)):
        distance += (v1[i] - v2[i]) ** 2
    return math.sqrt(distance)

def calculate_distances(test_weights, train_weights):
    """Calculate the distance between a single test weight and a list of train weights."""
    distances = [[(euclidean_distance(test_weight, train_weight), train_name, test_name) for train_weight, train_name in train_weights] 
                 for test_weight, test_name in test_weights]

    return distances 