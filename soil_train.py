import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import cv2
import numpy as np
import pickle

# Load the soil data
soil_data = pd.read_csv('soil.csv')  # Contains soil names and image paths
# Function to read and preprocess images
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))  # Resize images to 128x128 pixels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = image.flatten()  # Flatten the image
    return image

# Preprocess images and prepare the dataset
soil_data['image_array'] = soil_data['image_path'].apply(preprocess_image)
X = np.array(soil_data['image_array'].tolist())
y = soil_data['soil_name']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the trained model
with open('soil_classifier.pkl', 'wb') as f:
    pickle.dump(rf_model, f)