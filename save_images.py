import os
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from PIL import Image

def prepare_and_save_data(directory):
    # Fetch the dataset
    data = fetch_olivetti_faces()
    images = data.images
    targets = data.target
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, targets, test_size=0.25, random_state=42)
    
    # Save the training images
    train_dir = os.path.join(directory, 'train')
    os.makedirs(train_dir, exist_ok=True)
    for i, image in enumerate(X_train):
        img = Image.fromarray((image * 255).astype('uint8'))
        img.save(os.path.join(train_dir, f'train_{i}_label_{y_train[i]}.png'))

    # Save the testing images
    test_dir = os.path.join(directory, 'test')
    os.makedirs(test_dir, exist_ok=True)
    for i, image in enumerate(X_test):
        img = Image.fromarray((image * 255).astype('uint8'))
        img.save(os.path.join(test_dir, f'test_{i}_label_{y_test[i]}.png'))

    print("Images saved to", directory)

prepare_and_save_data('Olivetti')
