import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model
import pickle

# Load model
model = load_model("best_gru_model.keras")

# Load label encoder
with open("data/words_gru/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7
)

SEQ_LEN = 80
sequence = deque(maxlen=SEQ_LEN)
predictions = deque(maxlen=10)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    frame_landmarks = []

    if result.multi_hand_landmarks:

        hands_list = sorted(
            result.multi_hand_landmarks,
            key=lambda hand: hand.landmark[0].x
        )

        for i in range(2):
            if i < len(hands_list):

                hand = hands_list[i]
                wrist = hand.landmark[0]

                temp = []
                for lm in hand.landmark:
                    temp.extend([
                        lm.x - wrist.x,
                        lm.y - wrist.y,
                        lm.z - wrist.z
                    ])

                max_val = max(abs(v) for v in temp)
                if max_val != 0:
                    temp = [v / max_val for v in temp]

                frame_landmarks.extend(temp)

            else:
                frame_landmarks.extend([0.0] * 63)

    else:
        if sequence:
            frame_landmarks = sequence[-1]
        else:
            frame_landmarks = [0.0] * 126

    # ✅ INSIDE LOOP
    sequence.append(frame_landmarks)

    if len(sequence) == SEQ_LEN:

        input_data = np.array(sequence).reshape(1, SEQ_LEN, 126)

        pred = model.predict(input_data, verbose=0)[0]
        confidence = np.max(pred)
        class_id = np.argmax(pred)
        word = label_encoder.inverse_transform([class_id])[0]
        print(f"{word} | {confidence:.2f}")

        predictions.append(word)

    # ✅ INSIDE LOOP
    if len(predictions) > 0:
        final_word = max(set(predictions), key=predictions.count)
    else:
        final_word = ""

    # Display
    cv2.putText(frame, final_word, (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("ISL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()