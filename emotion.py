import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
from collections import Counter

# Load model and labels
model = load_model(r"\emotion_model2.h5")  #Replace with your location
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Emotion to Stress Mapping
stress_map = {
    'Neutral': 'no_stress',
    'Happy': 'no_stress',
    'Sad': 'mild_stress',
    'Surprise': 'mild_stress',
    'Fear': 'high_stress',
    'Angry': 'high_stress',
    'Disgust': 'high_stress'
}

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

# Timer setup
start_time = time.time()
detected_emotions = []
final_emotion = ""
final_stress = ""
status_text = "Detecting..."
first_detection_done = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    current_time = time.time()

    # First 5 seconds only once for "Detecting..."
    if not first_detection_done and current_time - start_time <= 5:
        status_text = "Detecting..."
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_img_resized = cv2.resize(face_img, (48, 48))
            face_img_norm = face_img_resized.astype("float32") / 255.0
            face_img_input = np.expand_dims(face_img_norm, axis=0)

            prediction = model.predict(face_img_input, verbose=0)
            emotion_index = np.argmax(prediction)
            emotion = emotion_labels[emotion_index]
            detected_emotions.append(emotion)

    # After first detection window
    elif not first_detection_done and current_time - start_time > 5:
        if detected_emotions:
            final_emotion = Counter(detected_emotions).most_common(1)[0][0]
            final_stress = stress_map[final_emotion]
            status_text = f"{final_emotion} ({final_stress})"
            print(f"Initial Detected: {final_emotion} -> {final_stress}")
        else:
            status_text = "No face detected"
        detected_emotions = []
        start_time = current_time
        first_detection_done = True

    # Every 5 seconds after initial detection
    elif first_detection_done and current_time - start_time <= 5:
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_img_resized = cv2.resize(face_img, (48, 48))
            face_img_norm = face_img_resized.astype("float32") / 255.0
            face_img_input = np.expand_dims(face_img_norm, axis=0)

            prediction = model.predict(face_img_input, verbose=0)
            emotion_index = np.argmax(prediction)
            emotion = emotion_labels[emotion_index]
            detected_emotions.append(emotion)

    elif first_detection_done and current_time - start_time > 5:
        if detected_emotions:
            final_emotion = Counter(detected_emotions).most_common(1)[0][0]
            final_stress = stress_map[final_emotion]
            status_text = f"{final_emotion} ({final_stress})"
            print(f"Updated: {final_emotion} -> {final_stress}")
        else:
            status_text = "No face detected"
        detected_emotions = []
        start_time = current_time

    # Draw face rectangles and label
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, status_text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Emotion and Stress Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
