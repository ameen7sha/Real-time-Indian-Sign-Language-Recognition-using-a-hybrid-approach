import streamlit as st
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
from collections import deque, Counter

gru_model = load_model("best_gru_model7.keras")
cnn_model = load_model("best7_cnn_model.keras")

with open("data7/words_grutrn/label_encoder.pkl", "rb") as f:
    gru_encoder = pickle.load(f)

with open("cnn_labels.pkl", "rb") as f:
    cnn_encoder = pickle.load(f)


mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)


st.set_page_config(page_title="ISL Recognition", layout="centered")
st.title("ISL Recognition System")

mode = st.radio(
    "Select Mode",
    ["GRU (Words)", "CNN (Alphabets)"]
)


class VideoProcessor(VideoTransformerBase):

    def __init__(self):

        # GRU uses 120 frames now
        self.sequence = deque(maxlen=120)

        # smoothing history
        self.pred_history = deque(maxlen=10)

    # ---------------------------------
    # LANDMARK EXTRACTION
    # ---------------------------------

    def extract_landmarks(self, frame):

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
            frame_landmarks = [0.0] * 126

        return frame_landmarks

    def predict_gru(self, frame):

        landmarks = self.extract_landmarks(frame)
        self.sequence.append(landmarks)

        if len(self.sequence) < 120:
            return "Collecting..."

        input_data = np.array(self.sequence).reshape(1, 120, 126)

        pred = gru_model.predict(input_data, verbose=0)[0]

        class_id = np.argmax(pred)
        conf = np.max(pred)

        if conf < 0.75:
            return "Detecting..."

        label = gru_encoder.inverse_transform([class_id])[0]

        self.pred_history.append(label)

        # majority vote smoothing
        most_common = Counter(self.pred_history).most_common(1)[0][0]

        return f"{most_common} ({conf:.2f})"

    def predict_cnn(self, frame):

        h, w, _ = frame.shape

        # center ROI box
        x1 = int(w * 0.30)
        y1 = int(h * 0.20)
        x2 = int(w * 0.70)
        y2 = int(h * 0.80)

        roi = frame[y1:y2, x1:x2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(gray, (64, 64))

        roi = roi / 255.0
        roi = roi.reshape(1, 64, 64, 1)

        pred = cnn_model.predict(roi, verbose=0)[0]

        class_id = np.argmax(pred)
        conf = np.max(pred)

        if conf < 0.75:
            return "Detecting..."

        label = cnn_encoder.inverse_transform([class_id])[0]

        return f"{label} ({conf:.2f})"

    def transform(self, frame):

        img = frame.to_ndarray(format="bgr24")

        h, w, _ = img.shape

        # draw ROI box
        cv2.rectangle(
            img,
            (int(w * 0.30), int(h * 0.20)),
            (int(w * 0.70), int(h * 0.80)),
            (0, 255, 0),
            2
        )

        if mode == "GRU (Words)":
            text = self.predict_gru(img)
        else:
            text = self.predict_cnn(img)

        cv2.putText(
            img,
            text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        return img

webrtc_streamer(
    key="isl",
    video_transformer_factory=VideoProcessor,
    media_stream_constraints={
        "video": {
            "width": 1280,
            "height": 720,
            "frameRate": 30
        },
        "audio": False
    }
)