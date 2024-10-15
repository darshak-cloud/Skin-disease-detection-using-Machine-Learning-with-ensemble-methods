Skin Disease Detection Using Machine Learning and Ensemble Methods
This project implements a machine learning system for detecting and classifying nine different skin diseases from images using ensemble methods to enhance accuracy.


Overview
The goal of this project is to build a model that can accurately classify nine types of skin diseases from images, leveraging ensemble learning techniques. The system extracts features from images using Histogram of Oriented Gradients (HOG) and trains multiple models combined into an ensemble to improve classification performance. The final model is deployed in a Flask web application, allowing users to either upload an image or capture one through a webcam for real-time prediction.


Dataset
The dataset used for training and validation is publicly available on Kaggle. It contains images for nine different skin diseases, each with 90-100 images.
•	Dataset: https://www.kaggle.com/datasets/riyaelizashaju/skin-disease-classification-image-dataset/data
Skin Diseases in the Dataset:
1.	Actinic Keratosis
2.	Atopic Dermatitis
3.	Benign Keratosis
4.	Dermatofibroma
5.	Melanocytic Nevus
6.	Melanoma
7.	Squamous Cell Carcinoma
8.	Tinea Ringworm Candidiasis
9.	Vascular Lesion


Key Features
•	Data Preparation: Preprocessing images for model training.
•	Feature Extraction: Using HOG to extract important features from images.
•	Ensemble Learning: Combining multiple models to improve classification accuracy.
•	Flask Deployment: Web application that allows image upload or live webcam input for skin disease detection.


Model Training
The ensemble learning approach includes training various classifiers to achieve an optimal accuracy of 99-100%.

Web Application
The web application is built using Flask and allows users to:
•	Upload an image of skin for disease prediction.
•	Use a webcam to capture live images for real-time detection.


Installation
To run this project locally:
1.	Clone this repository.
2.	Install the required dependencies from requirements.txt: pip install -r requirements.txt
3.	Run the Flask application: python app.py

File Structure
│folder name
├── app/
│   └── app.py               # Flask web application
├── templates/
│   ├── index.html           # Upload page
│   └── result.html          # Results page
├── static/
│   ├── style.css            # CSS styling
│   └── webcam.js            # Webcam functionality
├── models/
│   ├── # Trained ensemble model
├── scripts/
│   ├── data_preparation.py  # Data preparation script
│   ├── feature_extraction.py# HOG feature extraction
│   ├── model_training.py    # Model training script
│   └── model_evaluation.py  # Model evaluatio[Skin Disease Detection Using Machine Learning and Ensemble Methods.docx](https://github.com/user-attachments/files/17381767/Skin.Disease.Detection.Using.Machine.Learning.and.Ensemble.Methods.docx)
n script
└── uploads/                 # Folder for storing uploaded images


Usage
•	To run this project from script folder First you have to set up my environment variable if you are using VScode. If not there is no need.
•	For setting up myenv follow step: (make sure that you first delete myenv file that is already exist in the repository after copying or downloading)
1.	Open Terminal in VS Code: Use the integrated terminal in VS Code.
2.	Create a Virtual Environment: Run python -m venv myenv (replace myenv with your desired environment name).
3.	Activate the Virtual Environment:
Windows: .\myenv\Scripts\activate
macOS/Linux: source myenv/bin/activate
•	It looks like: (myenv) PS C:\Users\[your path]\GitHub\Skin-disease-detection-using-ML-with-ensemble-methods>
•	Now after training you just need to run app.py file.
•	Navigate to localhost:5000 in your web browser after running the Flask app.
•	Upload an image or use the webcam to get a skin disease prediction.


License
This project is not open source. All rights reserved. Please do not distribute or use this project without permission.

