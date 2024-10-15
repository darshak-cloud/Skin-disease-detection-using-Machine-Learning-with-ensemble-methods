import os
import sys
import joblib
from flask import Flask, render_template, request, redirect, flash, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import base64
import io
import numpy as np
import pickle  # Add this import

# Adjust this path if necessary
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

try:
    from feature_extraction import extract_hog_features
except ImportError as e:
    print(f"Error importing feature_extraction: {e}")
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__, template_folder=r'C:\Users\Asus\OneDrive\Desktop\s_detect\templates', 
            static_folder=r'C:\Users\Asus\OneDrive\Desktop\s_detect\static')
app.secret_key = "supersecretkey"

# Define paths and file locations
models_dir = r'C:\Users\Asus\OneDrive\Desktop\s_detect\models'
model_paths = {
    'adaboost': os.path.join(models_dir, 'adaboost_model.pkl'),
    'bagging': os.path.join(models_dir, 'bagging_model.pkl'),
    'extra_trees': os.path.join(models_dir, 'extra_trees_model.pkl'),
    'hist_gradient_boosting': os.path.join(models_dir, 'hist_gradient_boosting_model.pkl'),
    'feature_selector': os.path.join(models_dir, 'feature_selector.pkl'),
    'label_encoder': os.path.join(models_dir, 'label_encoder.pkl'),
}

# Load all models and feature selector
print("Loading models and feature selector...")
models = {}
try:
    for key in model_paths:
        models[key] = joblib.load(model_paths[key])
    print("Models and feature selector loaded successfully.")
except Exception as e:
    print(f"Error loading models or feature selector: {e}")
    sys.exit(1)

# Load validation data once globally
val_labels = np.load('val_labels.npy')
val_hog_features = pickle.load(open('C:/Users/Asus/OneDrive/Desktop/s_detect/val_hog_features.pkl', 'rb'))
X_val_selected = models['feature_selector'].transform(val_hog_features)

# Function to check allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to load validation data and evaluate model accuracies
def get_model_accuracies():
    accuracies = {}
    for name, model in models.items():
        if name in ['feature_selector', 'label_encoder']:
            continue
        predictions = model.predict(X_val_selected)
        accuracy = np.mean(predictions == val_labels)
        accuracies[name] = accuracy
    return accuracies

# Route for home page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files and 'image_data' not in request.form:
        flash('No file or image data provided')
        return redirect(request.url)

    file = request.files.get('file', None)
    image_data = request.form.get('image_data', None)
    
    file_path = None
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(r'C:\Users\Asus\OneDrive\Desktop\s_detect\uploads', filename)
        file.save(file_path)
        print(f"File saved to {file_path}")
    elif image_data:
        try:
            image_bytes = io.BytesIO(base64.b64decode(image_data.split(',')[1]))
            image = Image.open(image_bytes)
            file_path = os.path.join(r'C:\Users\Asus\OneDrive\Desktop\s_detect\uploads', 'captured_image.png')
            image.save(file_path)
            print(f"Captured image saved to {file_path}")
        except Exception as e:
            flash(f'Error processing webcam image: {str(e)}')
            return redirect(request.url)
    else:
        flash('No image uploaded or captured')
        return redirect(request.url)
    
    if file_path:
        try:
            # Extract HOG features from the image
            features = extract_hog_features(file_path)
            features = np.reshape(features, (1, -1))  # Ensure the feature array is in correct shape
            features_selected = models['feature_selector'].transform(features)
        except ValueError as e:
            flash(f'Feature extraction or selection failed: {e}')
            return redirect(request.url)

        try:
            # Initialize a dictionary to hold model names and their prediction accuracies
            model_accuracies = {}
            disease_name = None
            best_model_accuracy = 0
            best_model_name = None

            for model_name, model in models.items():
                if model_name in ['feature_selector', 'label_encoder']:  # Skip non-model objects
                    continue
                
                # Get the prediction
                prediction = model.predict(features_selected)[0]

                # Get the accuracy (probability) for the current model
                accuracy = max(model.predict_proba(features_selected)[0]) * 100
                model_accuracies[model_name] = accuracy

                # Update best model if this model has higher accuracy
                if accuracy > best_model_accuracy:
                    best_model_accuracy = accuracy
                    best_model_name = model_name
                    best_prediction = prediction

            # Mapping predictions to skin diseases
            disease_mapping = {
                0: 'Actinic Keratosis',
                1: 'Atopic Dermatitis',
                2: 'Benign Keratosis',
                3: 'Dermatofibroma',
                4: 'Melanocytic Nevus',
                5: 'Melanoma',
                6: 'Squamous Cell Carcinoma',
                7: 'Tinea Ringworm Candidiasis',
                8: 'Vascular Lesion'
            }

            # Map the best prediction to a disease name
            disease_name = disease_mapping.get(best_prediction, "Unknown Disease")
            print(f"Best model ({best_model_name}) predicted: {disease_name}")
            print(f"Prediction accuracies for all models: {model_accuracies}")

            # Return the result page with the prediction and accuracies
            return render_template(
                'result.html',
                prediction=disease_name,
                accuracy=best_model_accuracy,
                accuracies=model_accuracies,
                image_file=os.path.basename(file_path)
            )

        except Exception as e:
            print(f"Error during prediction: {e}")
            flash('Error during prediction. Please check the image file.')
            return redirect(request.url)
    else:
        flash('Error processing the image.')
        return redirect(request.url)

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(r'C:\Users\Asus\OneDrive\Desktop\s_detect\uploads', filename)

# Ensure uploads directory exists
if __name__ == '__main__':
    os.makedirs(r'C:\Users\Asus\OneDrive\Desktop\s_detect\uploads', exist_ok=True)
    app.run(debug=True)
