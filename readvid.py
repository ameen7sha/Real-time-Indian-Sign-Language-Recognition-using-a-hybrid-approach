import os
import cv2
import numpy as np
import mediapipe as mp

VIDEO_ROOT = "Videos/You"
SAVE_ROOT = "data7/words_landmarks/You"

# changed to 120
SEQ_LEN = 120

os.makedirs(SAVE_ROOT, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

print("VIDEO_ROOT:", VIDEO_ROOT)
print("Files found:", os.listdir(VIDEO_ROOT))

for video_file in os.listdir(VIDEO_ROOT):

    if not video_file.lower().endswith((".mp4", ".avi", ".mov")):
        continue

    video_path = os.path.join(VIDEO_ROOT, video_file)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Failed to open:", video_file)
        continue

    sequence = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        frame_landmarks = []

        if result.multi_hand_landmarks:

            # Keep left/right hand order consistent
            hands_list = sorted(
                result.multi_hand_landmarks,
                key=lambda hand: hand.landmark[0].x
            )

            for hand_idx in range(2):

                if hand_idx < len(hands_list):

                    hand_landmarks = hands_list[hand_idx]
                    wrist = hand_landmarks.landmark[0]

                    temp = []

                    for lm in hand_landmarks.landmark:
                        temp.extend([
                            lm.x - wrist.x,
                            lm.y - wrist.y,
                            lm.z - wrist.z
                        ])

                    # scale normalization
                    max_val = max(abs(val) for val in temp)

                    if max_val != 0:
                        temp = [val / max_val for val in temp]

                    frame_landmarks.extend(temp)

                else:
                    frame_landmarks.extend([0.0] * 63)

        else:
            # if no hand detected
            if sequence:
                frame_landmarks = sequence[-1]
            else:
                frame_landmarks = [0.0] * 126

        sequence.append(frame_landmarks)

    cap.release()

    if len(sequence) == 0:
        print("Empty sequence:", video_file)
        continue

    sequence = np.array(sequence)

    # -----------------------------------
    # 120 frames sampled across full clip
    # -----------------------------------
    if len(sequence) > SEQ_LEN:

        indices = np.linspace(
            0,
            len(sequence) - 1,
            SEQ_LEN
        ).astype(int)

        sequence = sequence[indices]

    else:
        padding = np.zeros((SEQ_LEN - len(sequence), 126))
        sequence = np.vstack((sequence, padding))

    save_name = video_file.rsplit(".", 1)[0] + ".npy"
    save_path = os.path.join(SAVE_ROOT, save_name)

    np.save(save_path, sequence)

    print("Saved:", save_name, "Shape:", sequence.shape)

hands.close()
print("All videos processed.")