# Matrices Project

This project involves the use of the Olivetti faces dataset to perform face recognition using Principal Component Analysis (PCA). The project saves images, converts them into matrices, computes mean faces, eigenfaces, and classifies test images based on their Euclidean distances from training images.

## Files

- **utils.py**: Contains utility functions for loading and processing images, creating matrices and vectors, calculating the mean face, covariance matrix, eigenvalues, eigenvectors, and weights, and computing Euclidean distances.
- **main.py**: The main script that utilizes functions from `utils.py` to perform face recognition.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/farahorfaly04/Matrices-project.git
    cd Matrices-project
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Main Script**:
    ```bash
    python main.py
    ```
    This script will:
    - Load and display the training and testing images.
    - Convert images into matrices and vectors.
    - Calculate and display the mean face image.
    - Compute the covariance matrix and perform PCA to obtain eigenfaces.
    - Calculate weights for the training and testing images.
    - Compute Euclidean distances between test and train weights and classify the test images based on these distances.

## Functions

### utils.py

- **load_image(image_path, size=(112,112))**: Loads and resizes an image.
- **create_matrices(folder_path)**: Converts images into matrices.
- **create_vectors(matrices)**: Converts matrices into vectors.
- **display_matrices(matrices)**: Displays matrices as images.
- **calculate_mean_face(vectors)**: Calculates the mean face image.
- **display_mean_face(mean_face_vector)**: Displays the mean face image.
- **calculate_covariance_matrix(vectors_matrix)**: Calculates the covariance matrix.
- **calculate_eigen(covariance_matrix)**: Computes eigenvalues and eigenvectors of the covariance matrix.
- **calculate_weights(eigenfaces, vectors_minus_mean)**: Calculates weights of the training images.
- **euclidean_distance(v1, v2)**: Calculates the Euclidean distance between two vectors.
- **calculate_distances(test_weights, train_weights)**: Computes distances between test weights and train weights.

## Results

The project displays the test images and their closest matching training images if the Euclidean distance is below a defined threshold. If no close match is found, the test image is classified as unknown.

