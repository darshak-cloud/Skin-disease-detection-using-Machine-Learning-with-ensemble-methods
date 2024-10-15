import pickle
import numpy as np
import joblib
from sklearn.feature_selection import SelectKBest, f_classif
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier



# Define paths for models and supporting files
model_paths = {
    'bagging': r'C:\Users\Asus\OneDrive\Desktop\s_detect\models\bagging_model.pkl',
    'adaboost': r'C:\Users\Asus\OneDrive\Desktop\s_detect\models\adaboost_model.pkl',
    'hist_gradient_boosting': r'C:\Users\Asus\OneDrive\Desktop\s_detect\models\hist_gradient_boosting_model.pkl',
    'extra_trees': r'C:\Users\Asus\OneDrive\Desktop\s_detect\models\extra_trees_model.pkl'
}
val_features_path = r'C:\Users\Asus\OneDrive\Desktop\s_detect\val_hog_features.pkl'
val_labels_path = r'C:\Users\Asus\OneDrive\Desktop\s_detect\val_labels.npy'
selector_path = r'C:\Users\Asus\OneDrive\Desktop\s_detect\models\feature_selector.pkl'
label_encoder_path = r'C:\Users\Asus\OneDrive\Desktop\s_detect\models\label_encoder.pkl'

# Load the feature selector
print(f"Loading feature selector from '{selector_path}'...")
try:
    with open(selector_path, 'rb') as f:
        selector = joblib.load(f)
    print("Feature selector loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Error loading feature selector: {e}")

# Load the label encoder
print(f"Loading label encoder from '{label_encoder_path}'...")
try:
    with open(label_encoder_path, 'rb') as f:
        label_encoder = joblib.load(f)
    print("Label encoder loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Error loading label encoder: {e}")

# Load validation data
print("Loading validation data...")
try:
    with open(val_features_path, 'rb') as f:
        X_val = pickle.load(f)
    y_val = np.load(val_labels_path)
    print("Validation data loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Error loading validation data: {e}")

# Convert list of arrays to a single array for validation
print("Preparing validation data...")
max_len = max(len(x) for x in X_val)
X_val = np.array([np.pad(x, (0, max_len - len(x)), 'constant') for x in X_val])

# Apply the saved feature selector
print("Applying feature selection...")
try:
    X_val_selected = selector.transform(X_val)
    print("Feature selection applied successfully.")
except Exception as e:
    raise RuntimeError(f"Error during feature selection: {e}")

# Load and evaluate each ensemble model
for model_name, model_path in model_paths.items():
    print(f"Loading and evaluating model: {model_name}...")
    try:
        # Load the trained model
        model = joblib.load(model_path)
        print(f"Model '{model_name}' loaded successfully.")
        
        # Make predictions
        y_pred = model.predict(X_val_selected)
        
        # Decode labels
        y_val_decoded = label_encoder.inverse_transform(y_val)
        y_pred_decoded = label_encoder.inverse_transform(y_pred)
        
        # Print evaluation metrics
        print(f"\n--- Evaluation for {model_name} ---")
        print("Classification Report:")
        print(classification_report(y_val_decoded, y_pred_decoded))
        
        # Confusion matrix
        print("Confusion Matrix:")
        conf_matrix = confusion_matrix(y_val_decoded, y_pred_decoded)
        print(conf_matrix)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.title(f"Confusion Matrix for {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()
        
        # Calculate and display accuracy
        accuracy = accuracy_score(y_val_decoded, y_pred_decoded)
        print(f"Accuracy for {model_name}: {accuracy:.2f}")
        
    except Exception as e:
        print(f"Error evaluating model '{model_name}': {e}")



#
#
# Visualizing the dataset distribution (using labels from validation data)
def plot_dataset_distribution(labels):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=labels)
    plt.title("Visualization of the Skin Disease Dataset")
    plt.xlabel("Skin Disease")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Call this function with your validation labels (e.g. `y_val` from .npy file)
plot_dataset_distribution(label_encoder.inverse_transform(y_val))



#
#
# Function to load HOG features from a .pkl file
def load_hog_features(file_path):
    with open(file_path, 'rb') as file:
        hog_features = pickle.load(file)
    return hog_features

# Define and train the model
def train_model(train_features, train_labels):
    # Define your classifiers
    pac = PassiveAggressiveClassifier()
    lda = LinearDiscriminantAnalysis()
    rnc = RadiusNeighborsClassifier(radius=100.0)  # Adjust radius if needed
    bnb = BernoulliNB()
    gnb = GaussianNB()
    etc = ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    bagging = BaggingClassifier(n_estimators=50, random_state=42)
    adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)
    hist_gradient_boosting = HistGradientBoostingClassifier(max_iter=100)

    # Combine classifiers into a voting classifier
    voting_clf = VotingClassifier(
        estimators=[
            ('pac', pac), ('lda', lda), ('rnc', rnc),
            ('bnb', bnb), ('gnb', gnb), ('etc', etc),
            ('bagging', bagging), ('adaboost', adaboost),
            ('hist_gradient_boosting', hist_gradient_boosting)
        ],
        voting='hard'
    )

    # Create a pipeline with feature scaling
    model = make_pipeline(StandardScaler(), voting_clf)
    
    # Train the model
    model.fit(train_features, train_labels)
    return model

# Function to calculate accuracy for training and validation data
def calculate_accuracies(model, train_features, train_labels, val_features, val_labels):
    # Training accuracy
    train_predictions = model.predict(train_features)
    train_accuracy = accuracy_score(train_labels, train_predictions)
    
    # Validation accuracy
    val_predictions = model.predict(val_features)
    val_accuracy = accuracy_score(val_labels, val_predictions)

    return train_accuracy, val_accuracy

# Function to plot the accuracies in percentage
def plot_accuracies(train_accuracy, val_accuracy):
    plt.figure(figsize=(10, 6))
    
    # Convert accuracies to percentages
    train_accuracy_percent = train_accuracy * 100
    val_accuracy_percent = val_accuracy * 100
    
    # Define the accuracies and labels
    accuracies = [train_accuracy_percent, val_accuracy_percent]
    labels = ['Training Accuracy', 'Validation Accuracy']
    
    # Bar plot for training and validation accuracy
    plt.bar(labels, accuracies, color=['blue', 'orange'])
    
    # Title and axis labels
    plt.title("Accuracy of the Model on Training and Validation Data")
    plt.xlabel("Dataset")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)  # Accuracy percentages range from 0 to 100
    plt.tight_layout()
    plt.show()

# Load the training and validation data
train_labels = np.load('train_labels.npy')
val_labels = np.load('val_labels.npy')

train_hog_features = load_hog_features('C:/Users/Asus/OneDrive/Desktop/s_detect/train_hog_features.pkl')
val_hog_features = load_hog_features('C:/Users/Asus/OneDrive/Desktop/s_detect/val_hog_features.pkl')

# Train the model
model = train_model(train_hog_features, train_labels)

# Calculate training and validation accuracies
train_accuracy, val_accuracy = calculate_accuracies(model, train_hog_features, train_labels, val_hog_features, val_labels)

# Plot the accuracies
plot_accuracies(train_accuracy, val_accuracy)



#
#
# Load HOG features and labels
def load_hog_features(file_path):
    with open(file_path, 'rb') as file:
        hog_features = pickle.load(file)
    return np.array(hog_features)  # Ensure features are returned as numpy array

# Function to check dimensions
def check_dimensions(features, labels):
    print(f"Features shape: {features.shape}")
    print(f"Labels length: {len(labels)}")
    if features.shape[0] != len(labels):
        raise ValueError("Mismatch between number of features and labels")

# Function to train a classifier and compute accuracy
def evaluate_classifier(clf, train_features, train_labels, val_features, val_labels):
    clf.fit(train_features, train_labels)
    val_predictions = clf.predict(val_features)
    val_accuracy = accuracy_score(val_labels, val_predictions)
    return val_accuracy

# Define and evaluate all classifiers
def calculate_classifiers_accuracies(train_features, train_labels, val_features, val_labels):
    # Check dimensions before proceeding
    check_dimensions(train_features, train_labels)
    check_dimensions(val_features, val_labels)
    
    # Define classifiers
    classifiers = {
        'PAC': PassiveAggressiveClassifier(),
        'LDA': LinearDiscriminantAnalysis(),
        'RNC': RadiusNeighborsClassifier(radius=100.0),
        'BNB': BernoulliNB(),
        'GNB': GaussianNB(),
        'ETC': ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    }
    
    accuracies = {}
    for name, clf in classifiers.items():
        accuracy = evaluate_classifier(clf, train_features, train_labels, val_features, val_labels)
        accuracies[name] = accuracy

    return classifiers.keys(), accuracies.values()

# Function to plot classifier accuracy in percentage
def plot_classifier_accuracy(classifiers, accuracies):
    plt.figure(figsize=(10, 6))
    
    # Convert accuracies to percentages
    accuracies_percent = [accuracy * 100 for accuracy in accuracies]
    
    # Bar plot for classifier accuracies
    sns.barplot(x=list(classifiers), y=accuracies_percent, palette='viridis')
    
    # Title and axis labels
    plt.title("Accuracy of Different Classifiers")
    plt.xlabel("Classifiers")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)  # Accuracy percentages range from 0 to 100
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Load the training and validation data
train_labels = np.load('C:/Users/Asus/OneDrive/Desktop/detect_skin/train_labels.npy')
val_labels = np.load('C:/Users/Asus/OneDrive/Desktop/detect_skin/val_labels.npy')

train_hog_features = load_hog_features('C:/Users/Asus/OneDrive/Desktop/detect_skin/train_hog_features.pkl')
val_hog_features = load_hog_features('C:/Users/Asus/OneDrive/Desktop/detect_skin/val_hog_features.pkl')

# Check if there is a mismatch in training and validation data
if len(train_labels) != len(train_hog_features) or len(val_labels) != len(val_hog_features):
    print("Mismatch detected in training or validation data.")
    # Find the indices for the first few examples to help debug
    print(f"Training features: {len(train_hog_features)}, Training labels: {len(train_labels)}")
    print(f"Validation features: {len(val_hog_features)}, Validation labels: {len(val_labels)}")
else:
    # Calculate accuracies if no mismatch
    classifiers, accuracies = calculate_classifiers_accuracies(train_hog_features, train_labels, val_hog_features, val_labels)

    # Plot the accuracies
    plot_classifier_accuracy(classifiers, accuracies)




#
#
# Define paths
val_labels_path = 'C:/Users/Asus/OneDrive/Desktop/s_detect/val_labels.npy'
val_hog_features_path = 'C:/Users/Asus/OneDrive/Desktop/s_detect/val_hog_features.pkl'
models_dir = 'C:/Users/Asus/OneDrive/Desktop/s_detect/models'
feature_selector_path = os.path.join(models_dir, 'feature_selector.pkl')

# Load HOG features and labels
def load_hog_features(file_path):
    with open(file_path, 'rb') as file:
        hog_features = pickle.load(file)
    return np.array(hog_features)  # Ensure features are returned as numpy array

# Apply feature selection
def apply_feature_selection(X, selector):
    return selector.transform(X)

# Function to calculate accuracy of a model
def calculate_model_accuracy(model, val_features, val_labels):
    val_predictions = model.predict(val_features)
    accuracy = accuracy_score(val_labels, val_predictions)
    return accuracy

# Function to plot model accuracy
def plot_model_accuracy(model_name, accuracy):
    plt.figure(figsize=(10, 6))
    
    # Convert accuracy to percentage
    accuracy_percent = accuracy * 100
    
    # Bar plot for model accuracy
    sns.barplot(x=[model_name], y=[accuracy_percent], palette='viridis')
    
    # Title and axis labels
    plt.title(f"Accuracy of {model_name} Model")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)  # Accuracy percentages range from 0 to 100
    plt.tight_layout()
    plt.show()

# Load validation data
val_labels = np.load(val_labels_path)
val_hog_features = load_hog_features(val_hog_features_path)

# Load the feature selector
selector = joblib.load(feature_selector_path)

# Apply feature selection to validation data
X_val_selected = apply_feature_selection(val_hog_features, selector)

# Define and load ensemble models
ensemble_model_names = ['bagging_model.pkl', 'adaboost_model.pkl', 'hist_gradient_boosting_model.pkl', 'extra_trees_model.pkl']
ensemble_models = {name: joblib.load(os.path.join(models_dir, name)) for name in ensemble_model_names}

# Evaluate and plot accuracy for each model
for model_name, model in ensemble_models.items():
    print(f"Evaluating {model_name}...")
    accuracy = calculate_model_accuracy(model, X_val_selected, val_labels)
    print(f"{model_name} Accuracy: {accuracy:.2f}")
    plot_model_accuracy(model_name.replace('_model.pkl', '').replace('_', ' ').title(), accuracy)  # Format model name for plotting



#
#
# Define paths
val_labels_path = 'C:/Users/Asus/OneDrive/Desktop/s_detect/val_labels.npy'
val_hog_features_path = 'C:/Users/Asus/OneDrive/Desktop/s_detect/val_hog_features.pkl'
models_dir = 'C:/Users/Asus/OneDrive/Desktop/s_detect/models'
feature_selector_path = os.path.join(models_dir, 'feature_selector.pkl')

# Define class names
class_names = ['Actinic keratosis', 'Atopic Dermatitis', 'Benign keratosis', 'Dermatofibroma',
                'Melanocytic nevus', 'Melanoma', 'Squamous cell carcinoma', 'Tinea Ringworm Candidiasis',
                'Vascular lesion']

# Load HOG features and labels
def load_hog_features(file_path):
    with open(file_path, 'rb') as file:
        hog_features = pickle.load(file)
    return np.array(hog_features)  # Ensure features are returned as numpy array

# Apply feature selection
def apply_feature_selection(X, selector):
    return selector.transform(X)

# Function to calculate accuracy and confusion matrix of a model
def evaluate_model(model, val_features, val_labels):
    val_predictions = model.predict(val_features)
    accuracy = accuracy_score(val_labels, val_predictions)
    conf_matrix = confusion_matrix(val_labels, val_predictions)
    return accuracy, conf_matrix

# Function to plot confusion matrix
def plot_confusion_matrix(conf_matrix, class_names):
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix Visualization")
    plt.tight_layout()
    plt.show()

# Load validation data
val_labels = np.load(val_labels_path)
val_hog_features = load_hog_features(val_hog_features_path)

# Load the feature selector
selector = joblib.load(feature_selector_path)

# Apply feature selection to validation data
X_val_selected = apply_feature_selection(val_hog_features, selector)

# Define and load the models
model_files = [
    'bagging_model.pkl', 
    'adaboost_model.pkl', 
    'hist_gradient_boosting_model.pkl', 
    'extra_trees_model.pkl'
]

# Load models and evaluate
for model_file in model_files:
    model_path = os.path.join(models_dir, model_file)
    model = joblib.load(model_path)
    print(f"Evaluating {model_file}...")
    accuracy, conf_matrix = evaluate_model(model, X_val_selected, val_labels)
    accuracy_percentage = accuracy * 100
    print(f"{model_file} Accuracy: {accuracy_percentage:.2f}%")
    plot_confusion_matrix(conf_matrix, class_names)
