from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
            matrices.append(matrix)

    return matrices

def create_vectors(matrices):
    """Convert the matrices into vectors """
    vectors = []
    for matrix in matrices:
        vector = matrix.flatten().reshape(-1, 1)
        vectors.append(vector)

    return vectors

def display_matrix(matrix):
    """Display a single matrix"""
    print("Shape of train_matrices_array:", matrix.shape)
    print("Ben Affleck 1 Matrix (112x112):")
    print(matrix) 

def display_matrices(matrices):
    """Display matrices as images"""
    num_images = len(matrices)
    num_cols = 5  # Number of columns in the display grid
    num_rows = (num_images + num_cols - 1) // num_cols  # Calculate number of rows based on number of images and columns

    plt.figure(figsize=(15, 3*num_rows))  # Adjust figsize based on number of rows

    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(matrices[i], cmap='gray')
        plt.title(f'Image {i+1}')
        plt.axis('off')

    plt.show()

def calculate_mean_face(vectors):
    """ Calculate the mean face image"""
    # Concatenate all vectors into a single matrix
    vectors_matrix = np.concatenate(vectors, axis=1)

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
    n_components = 20

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

    # List to store weights for each image
    weights_list = []

    eigenfaces_reshaped = [(eigenvector.flatten().reshape(-1,1)).T for eigenvector in eigenfaces]
  
    # Calculate weights for each image
    for i in range(len(vectors_minus_mean) - 1):
        
        # Calculate the weights for the image
        weights = np.dot(eigenfaces_reshaped[i], vectors_minus_mean[i])

        # Append weights to the list
        weights_list.append(weights)

    return eigenfaces_reshaped, weights_list

