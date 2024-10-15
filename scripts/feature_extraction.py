import cv2
import numpy as np
import os
import pickle
from skimage.feature import hog
from sklearn.feature_selection import SelectKBest, f_classif

# Define directories (Update these paths according to your dataset in Google Drive)
train_dir = r'C:\Users\Asus\OneDrive\Desktop\skin disease dataset\archive\Split_smol\train'
val_dir = r'C:\Users\Asus\OneDrive\Desktop\skin disease dataset\archive\Split_smol\val'
features_dir = r'C:\Users\Asus\OneDrive\Desktop\s_detect'

def extract_hog_features(image_path, feature_size=2000):
    """Extract HOG features from an image and ensure consistent feature size."""
    # Read the image
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Image cannot be loaded from path: {image_path}")
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Extract HOG features optimized for TPU
    hog_features = hog(
        gray_image, 
        orientations=9,  # Increased orientations for better feature capture
        pixels_per_cell=(16, 16), 
        cells_per_block=(2, 2), 
        block_norm='L2-Hys', 
        visualize=False  # Skip visualization for faster processing
    )
    
    # Pad or truncate features to ensure consistent length
    if len(hog_features) < feature_size:
        hog_features = np.pad(hog_features, (0, feature_size - len(hog_features)), 'constant')
    elif len(hog_features) > feature_size:
        hog_features = hog_features[:feature_size]
    
    # Print feature information for verification
    print(f"Extracted HOG features shape: {hog_features.shape}")
    print(f"First 5 feature values: {hog_features[:5]}")

    return hog_features

def process_images(directory, selector=None):
    """Process images in a directory to extract HOG features and labels."""
    features_list = []
    labels_list = []
    feature_size = 2000  # Adjust to match your SelectKBest setting

    for label, disease in enumerate(diseases):
        disease_dir = os.path.join(directory, disease)
        if os.path.isdir(disease_dir):
            print(f"Processing images in {disease_dir}")
            for filename in os.listdir(disease_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(disease_dir, filename)
                    try:
                        features = extract_hog_features(image_path, feature_size=feature_size)
                        features_list.append(features)
                        labels_list.append(label)
                    except ValueError as e:
                        print(f"Error processing {image_path}: {e}")
        else:
            print(f"Directory not found: {disease_dir}")

    features_list = np.array(features_list)
    if selector is not None:
        features_list = selector.transform(features_list)
    
    labels_list = np.array(labels_list) 
    return features_list, labels_list

def save_features_and_labels(features, labels, features_file, labels_file):
    """Save features and labels to files."""
    with open(features_file, 'wb') as f:
        pickle.dump(features, f)
    np.save(labels_file, np.array(labels))

# List of diseases
diseases = [
    'Actinic keratosis', 'Atopic Dermatitis', 'Benign keratosis', 'Dermatofibroma',
    'Melanocytic nevus', 'Melanoma', 'Squamous cell carcinoma',
    'Tinea Ringworm Candidiasis', 'Vascular lesion'
]

# Process and save training images
print("Processing training images...")
X_train, y_train = process_images(train_dir)

# Dimensionality reduction using SelectKBest
print("Applying SelectKBest to training data...")
k_best = 2000  # Number of features to select
selector = SelectKBest(f_classif, k=k_best)
X_train_selected = selector.fit_transform(X_train, y_train)

print("Saving selected training features and labels...")
save_features_and_labels(X_train_selected, y_train, os.path.join(features_dir, 'train_hog_features.pkl'), os.path.join(features_dir, 'train_labels.npy'))

# Process and save validation images
print("Processing validation images...")
X_val, y_val = process_images(val_dir, selector=selector)

print("Saving selected validation features and labels...")
save_features_and_labels(X_val, y_val, os.path.join(features_dir, 'val_hog_features.pkl'), os.path.join(features_dir, 'val_labels.npy'))

print("Feature extraction and dimensionality reduction completed.")