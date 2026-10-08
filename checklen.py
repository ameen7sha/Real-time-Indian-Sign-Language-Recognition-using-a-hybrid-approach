import os
import cv2

VIDEO_ROOT = "ISL"   # root folder containing all word folders

frame_counts = []

for word in os.listdir(VIDEO_ROOT):

    word_path = os.path.join(VIDEO_ROOT, word)

    if not os.path.isdir(word_path):
        continue

    for video in os.listdir(word_path):

        if not video.endswith((".mp4",".avi",".mov")):
            continue

        video_path = os.path.join(word_path, video)

        cap = cv2.VideoCapture(video_path)

        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_counts.append(frames)

        cap.release()


print("Total videos:", len(frame_counts))
print("Min frames:", min(frame_counts))
print("Max frames:", max(frame_counts))
print("Average frames:", sum(frame_counts) / len(frame_counts))