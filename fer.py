import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# Load dataset
data = pd.read_csv("Data/fertilizer.csv")

# Features and target variable
X = data[['N', 'P', 'K']]  # Input features: Nitrogen, Phosphorus, Potassium
y = data['Crop']  # Target: Crop name

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with scaling and SVM
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features by removing mean and scaling to unit variance
    ('svm', SVC(kernel='linear', random_state=42))  # Use linear kernel for SVM
])
# Train the model
pipeline.fit(X_train, y_train)

# Save the model
with open("fertilizer_model.pkl", "wb") as file:
    pickle.dump(pipeline, file)

print("Model trained and saved as 'fertilizer_model.pkl'.")