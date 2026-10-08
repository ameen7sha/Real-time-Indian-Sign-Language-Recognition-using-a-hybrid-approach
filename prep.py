import os
import cv2
import numpy as np

DATASET_PATH = "Alphabets"
IMG_SIZE = 64

#Load
images = []
labels = []

for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(label_path):
        continue

    for img_file in os.listdir(label_path):
        img_path = os.path.join(label_path, img_file)

        img = cv2.imread(img_path)
        if img is None:
            continue

        images.append(img)
        labels.append(label)

print("Step 1 done")
print("Images loaded:", len(images))
print("Labels loaded:", len(labels))

#Resize
resized_images = []

for img in images:
    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    resized_images.append(resized)

resized_images = np.array(resized_images)

print("Step 2 done")
print("Resized images shape:", resized_images.shape)

