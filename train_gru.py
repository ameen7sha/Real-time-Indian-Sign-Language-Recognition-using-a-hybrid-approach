import numpy as np
import pickle
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    GRU, Dense, Dropout,
    Bidirectional, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# -----------------------------------
# LOAD DATA
# -----------------------------------

DATA_DIR = "data7/words_grutrn"

X_train = np.load(f"{DATA_DIR}/X_train.npy")
X_val   = np.load(f"{DATA_DIR}/X_val.npy")

y_train = np.load(f"{DATA_DIR}/y_train.npy")
y_val   = np.load(f"{DATA_DIR}/y_val.npy")

with open(f"{DATA_DIR}/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Classes:", y_train.shape[1])

# -----------------------------------
# MODEL
# -----------------------------------

model = Sequential([

    Bidirectional(
        GRU(96, return_sequences=True),
        input_shape=(120,126)
    ),
    Dropout(0.4),

    Bidirectional(
        GRU(48)
    ),
    Dropout(0.4),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(y_train.shape[1], activation='softmax')
])

# -----------------------------------
# COMPILE
# -----------------------------------

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss=tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=0.05
    ),
    metrics=['accuracy']
)

model.summary()

# -----------------------------------
# CALLBACKS
# -----------------------------------

callbacks = [

    EarlyStopping(
        monitor='val_accuracy',
        patience=6,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-5,
        verbose=1
    ),

    ModelCheckpoint(
        "best_gru_model7.keras",
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# -----------------------------------
# TRAIN
# -----------------------------------

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=callbacks
)

# -----------------------------------
# SAVE FINAL
# -----------------------------------

model.save("final_gru_model7.keras")

print("GRU training complete.")