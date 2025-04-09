import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import Counter
import time
import paho.mqtt.client as mqtt

# Load model and labels
model = load_model(r"C:\Users\aswin\OneDrive\Desktop\New folder\emotion_model2.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Emotion to Stress Mapping
stress_map = {
    'Neutral': 'no_stress',
    'Happy': 'no_stress',
    'Sad': 'mild_stress',
    'Surprise': 'no_stress',
    'Fear': 'high_stress',
    'Angry': 'high_stress',
    'Disgust': 'high_stress'
}

# MQTT Setup
MQTT_BROKER = "192.168.215.225"  # or your broker IP
MQTT_PORT = 1883
MQTT_TOPIC = "test/topic"
client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Webcam
cap = cv2.VideoCapture(0)

# State variables
start_time = time.time()
detected_emotions = []
first_detection_done = False
status_text = "Detecting..."
stress_history = []
message_sent = False  # To avoid sending after high_stress

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    current_time = time.time()

    if not first_detection_done and current_time - start_time <= 5:
        status_text = "Detecting..."
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_img_resized = cv2.resize(face_img, (48, 48))
            face_img_norm = face_img_resized.astype("float32") / 255.0
            face_img_input = np.expand_dims(face_img_norm, axis=0)
            prediction = model.predict(face_img_input, verbose=0)
            emotion = emotion_labels[np.argmax(prediction)]
            detected_emotions.append(emotion)

    elif not first_detection_done and current_time - start_time > 5:
        if detected_emotions:
            final_emotion = Counter(detected_emotions).most_common(1)[0][0]
            final_stress = stress_map[final_emotion]
            stress_history.append(final_stress)
            status_text = f"({final_stress})"
            print(f"Initial Detected:  {final_stress}")
        else:
            status_text = "No face detected"
        detected_emotions = []
        start_time = current_time
        first_detection_done = True

    elif first_detection_done and current_time - start_time <= 5:
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_img_resized = cv2.resize(face_img, (48, 48))
            face_img_norm = face_img_resized.astype("float32") / 255.0
            face_img_input = np.expand_dims(face_img_norm, axis=0)
            prediction = model.predict(face_img_input, verbose=0)
            emotion = emotion_labels[np.argmax(prediction)]
            detected_emotions.append(emotion)

    elif first_detection_done and current_time - start_time > 5:
        if detected_emotions:
            final_emotion = Counter(detected_emotions).most_common(1)[0][0]
            final_stress = stress_map[final_emotion]
            stress_history.append(final_stress)
            print(f"Updated:  {final_stress}")
            status_text = f" ({final_stress})"

            # Check for 3 consecutive stress detections and message not yet sent
            if not message_sent:
                if len(stress_history) >= 3 and all(s == 'high_stress' for s in stress_history[-3:]):
                    client.publish(MQTT_TOPIC, "1")
                    print("MQTT Message Sent: 1 (high_stress)")
                    message_sent = True
                elif len(stress_history) >= 3 and all(s == 'mild_stress' for s in stress_history[-3:]):
                    client.publish(MQTT_TOPIC, "0")
                    print("MQTT Message Sent: 0 (mild_stress)")
        else:
            status_text = "No face detected"
        detected_emotions = []
        start_time = current_time

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, status_text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Emotion and Stress Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
client.disconnect()
