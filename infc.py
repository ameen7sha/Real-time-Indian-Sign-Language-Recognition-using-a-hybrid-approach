import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
import pickle

# Load model
model = load_model("cnn_model.h5")

# Load label encoder
with open("data/alphabets_cnn/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Webcam
cap = cv2.VideoCapture(0)

# Buffers
predictions = deque(maxlen=10)

CONF_THRESHOLD = 0.8

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔥 Preprocess (MUST match training)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(gray, (64, 64))
    roi = roi / 255.0
    roi = roi.reshape(1, 64, 64, 1)

    # 🔥 Prediction
    pred = model.predict(roi, verbose=0)[0]
    confidence = np.max(pred)
    class_id = np.argmax(pred)
    letter = label_encoder.inverse_transform([class_id])[0]

    print(f"{letter} | {confidence:.2f}")

    # 🔥 Confidence filter
    if confidence > CONF_THRESHOLD:
        predictions.append(letter)

    # 🔥 Stabilization
    final_letter = ""

    if len(predictions) == predictions.maxlen:
        most_common = max(set(predictions), key=predictions.count)

        if predictions.count(most_common) > 6:
            final_letter = most_common

    # 🔥 Display
    cv2.putText(frame, final_letter, (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow("CNN Alphabet Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()