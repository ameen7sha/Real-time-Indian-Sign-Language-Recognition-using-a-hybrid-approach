import numpy as np
import pickle

DATA_DIR = "data/alphabets_cnn"

X_train = np.load(f"{DATA_DIR}/X_train.npy")
X_val = np.load(f"{DATA_DIR}/X_val.npy")
y_train = np.load(f"{DATA_DIR}/y_train.npy")
y_val = np.load(f"{DATA_DIR}/y_val.npy")

with open(f"{DATA_DIR}/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)
print("Number of classes:", y_train.shape[1])

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from tensorflow.keras.callbacks import EarlyStopping

model = Sequential([

    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(y_train.shape[1], activation='softmax')

])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=32,
    callbacks=[early_stop]
)

model.save("cnn_model.h5")

print("CNN model saved successfully.")