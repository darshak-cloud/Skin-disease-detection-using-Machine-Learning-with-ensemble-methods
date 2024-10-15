from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_hist_gradient_boosting  # Required for HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
import numpy as np
import pickle
import joblib
import os

# Define paths and file locations
train_features_path = r'C:\Users\Asus\OneDrive\Desktop\s_detect\train_hog_features.pkl'  # Path to training HOG features
train_labels_path = r'C:\Users\Asus\OneDrive\Desktop\s_detect\train_labels.npy'  # Path to training labels
models_dir = r'C:\Users\Asus\OneDrive\Desktop\s_detect\models'  # Directory where the trained models will be saved

# Define the list of skin diseases
diseases = [
    'Actinic keratosis', 'Atopic Dermatitis', 'Benign keratosis', 'Dermatofibroma',
    'Melanocytic nevus', 'Melanoma', 'Squamous cell carcinoma',
    'Tinea Ringworm Candidiasis', 'Vascular lesion'
]

# Create a label encoder for the diseases
label_encoder = LabelEncoder()
label_encoder.fit(diseases)

# Save the label encoder for future use
label_encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
with open(label_encoder_path, 'wb') as f:
    joblib.dump(label_encoder, f)

# Load HOG features and labels
print("Loading training data...")
try:
    with open(train_features_path, 'rb') as f:
        X_train = pickle.load(f)
    y_train = np.load(train_labels_path)
    print("Training data loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Error loading data: {e}")

# Check the unique values in y_train
print("Unique values in y_train before encoding:", np.unique(y_train))

# Convert list of arrays to a single array for model training
print("Preparing training data...")
# Find the maximum length of HOG feature vectors
max_len = max(len(x) for x in X_train)
# Pad features to ensure consistent dimension
X_train = np.array([np.pad(x, (0, max_len - len(x)), 'constant') for x in X_train])

# Check if y_train contains integer labels (0, 1, 2, ...) and use them directly if so
if np.issubdtype(y_train.dtype, np.integer):
    y_train_encoded = y_train
else:
    # Encode labels to numerical values
    try:
        y_train_encoded = label_encoder.transform(y_train)
    except ValueError as e:
        raise RuntimeError(f"Label encoding error: {e}")

# Feature selection to reduce the number of features
print("Applying feature selection...")
k_best = 2000  # Adjust based on memory constraints
selector = SelectKBest(f_classif, k=k_best)
X_train_selected = selector.fit_transform(X_train, y_train_encoded)

# Verify the shape of the selected features
print(f"Selected feature shape: {X_train_selected.shape}")

# Save the feature selector
selector_path = os.path.join(models_dir, 'feature_selector.pkl')
with open(selector_path, 'wb') as f:
    joblib.dump(selector, f)
print(f"Feature selector saved to {selector_path}")

# Define individual classifiers
pac = PassiveAggressiveClassifier()
lda = LinearDiscriminantAnalysis()
rnc = RadiusNeighborsClassifier(radius=100.0)  # Adjust radius if needed
bnb = BernoulliNB()
gnb = GaussianNB()
extra_trees = ExtraTreesClassifier(n_estimators=100, random_state=42)  # Added ExtraTreesClassifier

# Define ensemble methods
# Using HistGradientBoostingClassifier for faster and TPU-optimized gradient boosting
hist_gradient_boosting = HistGradientBoostingClassifier(max_iter=100)

# Original ensemble models
ensemble_models = {
    'bagging': BaggingClassifier(n_estimators=50, random_state=42),
    'adaboost': AdaBoostClassifier(n_estimators=50, random_state=42),
    'hist_gradient_boosting': hist_gradient_boosting,
    'extra_trees': extra_trees  # Added ExtraTreesClassifier
}

# Train and evaluate each ensemble model
for name, model in ensemble_models.items():
    print(f"Training {name} model...")
    try:
        # Training the model
        model.fit(X_train_selected, y_train_encoded)
        print(f"{name} model trained successfully.")

        # Save the trained model
        model_path = os.path.join(models_dir, f'{name}_model.pkl')
        joblib.dump(model, model_path)
        print(f"{name} model saved to {model_path}")

        # Predict on training data for evaluation
        y_train_pred = model.predict(X_train_selected)

        # Evaluate the model
        accuracy = accuracy_score(y_train_encoded, y_train_pred)
        conf_matrix = confusion_matrix(y_train_encoded, y_train_pred)
        report = classification_report(y_train_encoded, y_train_pred, target_names=diseases)

        print(f"{name} model evaluation:")
        print(f"Accuracy: {accuracy:.2f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(report)

    except ValueError as e:
        print(f"ValueError during {name} model training: {e}")
    except Exception as e:
        print(f"Exception during {name} model training: {e}")

print("All models trained and evaluated.")
