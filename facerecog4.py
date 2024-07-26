import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import shutil
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2

# Paths to datasets
faces_data_dir = pathlib.Path('Faces')

# Pre-trained FaceNet model (Make sure you have this model available)
facenet_model = load_model('facenet_keras.h5')

# Function to preprocess image for FaceNet
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (160, 160))
    img = img.astype('float32')
    mean, std = img.mean(), img.std()
    img = (img - mean) / std
    img = np.expand_dims(img, axis=0)
    return img

# Extract embeddings
def get_embeddings(image_path):
    img = preprocess_image(image_path)
    embeddings = facenet_model.predict(img)
    return embeddings[0]

# Load dataset and extract embeddings
data_dir = faces_data_dir
X = []
y = []
for person_dir in data_dir.iterdir():
    if person_dir.is_dir():
        label = person_dir.name
        for image_path in person_dir.glob('*.jpg'):
            embeddings = get_embeddings(str(image_path))
            X.append(embeddings)
            y.append(label)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=123)

# Train SVM classifier
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

# Evaluate classifier
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Function to predict person's identity
def predict_person(image_path):
    embeddings = get_embeddings(image_path)
    predictions = svm_model.predict_proba([embeddings])[0]
    predicted_index = np.argmax(predictions)
    predicted_class = label_encoder.inverse_transform([predicted_index])[0]
    confidence = predictions[predicted_index]
    
    plt.figure()
    img = plt.imread(image_path)
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class} ({confidence * 100:.2f}%)")
    plt.axis("off")
    plt.show()

    print(f"Predicted: {predicted_class} with confidence: {confidence * 100:.2f}%")

# Example usage
image_path = 'path/to/new_face_image.jpg'
predict_person(image_path)
