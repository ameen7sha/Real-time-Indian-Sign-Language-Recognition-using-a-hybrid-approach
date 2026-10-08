import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# -------------------------
# LOAD MODEL
# -------------------------
model = load_model("best7_cnn_model.keras")   # or .keras if needed

with open("cnn_labels.pkl", "rb") as f:
    encoder = pickle.load(f)

# if saved as dict from class_indices
if isinstance(encoder, dict):
    labels = {v: k for k, v in encoder.items()}
else:
    labels = None

# -------------------------
# CAMERA
# -------------------------
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

frame_count = 0
last_text = "Starting..."

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    # ROI box
    x1, y1 = 200, 100
    x2, y2 = 450, 350

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    roi = frame[y1:y2, x1:x2]

    # Predict every 5 frames only (important)
    if frame_count % 5 == 0:

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(gray, (64,64))
        img = img / 255.0
        img = img.reshape(1,64,64,1)

        pred = model.predict(img, verbose=0)[0]

        idx = np.argmax(pred)
        conf = np.max(pred)

        if conf > 0.75:
            if labels:
                last_text = f"{labels[idx]} ({conf:.2f})"
            else:
                last_text = f"Class {idx} ({conf:.2f})"
        else:
            last_text = "Detecting..."

    frame_count += 1

    cv2.putText(frame, last_text, (30,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,0), 2)

    cv2.imshow("CNN Alphabet Recognition", frame)

    key = cv2.waitKey(1)
    if key == 27:   # ESC
        break

cap.release()
cv2.destroyAllWindows()