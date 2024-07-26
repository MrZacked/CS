import numpy as np
import cv2
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras_facenet import FaceNet
import joblib
import time


faces_data_dir = Path('Faces')


embedder = FaceNet()


def get_embeddings(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    detections = embedder.extract(img, threshold=0.95)
    if detections:
        return detections[0]['embedding']
    else:
        return None


def load_dataset_and_extract_embeddings(data_dir):
    data_dir = Path(data_dir)
    X, y = [], []
    for person_dir in data_dir.iterdir():
        if person_dir.is_dir():
            label = person_dir.name
            for image_path in person_dir.glob('*.jpg'):
                img = cv2.imread(str(image_path))
                if img is None:
                    continue
                embeddings = get_embeddings(img)
                if embeddings is not None:
                    X.append(embeddings)
                    y.append(label)
    return np.array(X), np.array(y)


X, y = load_dataset_and_extract_embeddings(faces_data_dir)


if X.size == 0 or y.size == 0:
    print("There was an issue with the embedded data set.")
else:

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=123)

    
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X_train, y_train)

    
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f" Accuray of the model: {accuracy * 100:.2f}%")

    
    joblib.dump(svm_model, 'svm_face_recognition_model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')


def real_time_face_recognition():
    cap = cv2.VideoCapture(0)  

    
    if not cap.isOpened():
        print("There was an error with opening the webcam.")
        return

    svm_model = joblib.load('svm_face_recognition_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')

    name_change_threshold = 60
    last_detected_name = None
    name_change_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Couldn't find the frame")
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = embedder.extract(img_rgb, threshold=0.95)

        for detection in detections:
            x1, y1, width, height = detection['box']
            x2, y2 = x1 + width, y1 + height

            
            embedding = detection['embedding']

            
            predictions = svm_model.predict_proba([embedding])[0]
            predicted_index = np.argmax(predictions)
            predicted_class = label_encoder.inverse_transform([predicted_index])[0]
            confidence = predictions[predicted_index]

            current_time = time.time()

            
            if last_detected_name is None or predicted_class != last_detected_name:
                if current_time - name_change_time < name_change_threshold:
                    predicted_class = "Deniz Utku Ates"
                else:
                    name_change_time = current_time

            last_detected_name = predicted_class

            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{predicted_class} ({confidence * 100:.2f}%)" if predicted_class != "Unknown" else "Unknown"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        
        cv2.imshow('RTF by DUA', frame)

    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


real_time_face_recognition()
