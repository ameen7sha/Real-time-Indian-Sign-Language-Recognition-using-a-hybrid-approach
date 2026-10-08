import os
import numpy as np
import pickle

LANDMARK_ROOT = "data7\words_landmarks"
SEQ_LEN = 120
FEATURES = 126

X = []
y = []


for word in os.listdir(LANDMARK_ROOT):

    word_path = os.path.join(LANDMARK_ROOT, word)

    if not os.path.isdir(word_path):
        continue

    for file in os.listdir(word_path):

        if file.endswith(".npy"):

            file_path = os.path.join(word_path, file)

            data = np.load(file_path)

        
            assert data.shape == (SEQ_LEN, FEATURES), \
                f"Wrong shape in {file}: {data.shape}"

            X.append(data)
            y.append(word)

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("STEP A DONE")
print("Original X shape:", X.shape)   
print("Original y shape:", y.shape)



def augment_sequence(seq):
    s = seq.copy()

    # small gaussian noise
    s += np.random.normal(0, 0.01, s.shape)

    # repeat some frames (tracking jitter simulation)
    idx = np.random.randint(1, len(s)-1, 5)

    for i in idx:
        s[i] = s[i - 1]

    # clip values
    s = np.clip(s, -1.5, 1.5)

    return s.astype(np.float32)


X_aug = []
y_aug = []

for i in range(len(X)):

    # original sample
    X_aug.append(X[i])
    y_aug.append(y[i])

    # augmented copy 1
    X_aug.append(augment_sequence(X[i]))
    y_aug.append(y[i])

    # augmented copy 2
    X_aug.append(augment_sequence(X[i]))
    y_aug.append(y[i])

X = np.array(X_aug, dtype=np.float32)
y = np.array(y_aug)

print("\nSTEP A.5 DONE")
print("Augmented X shape:", X.shape)
print("Augmented y shape:", y.shape)

# -------------------------------------------------
# STEP B: LABEL ENCODING
# -------------------------------------------------

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded)

print("\nSTEP B DONE")
print("Classes:", label_encoder.classes_)
print("Number of classes:", y_onehot.shape[1])
print("Encoded shape:", y_encoded.shape)
print("One-hot shape:", y_onehot.shape)

# -------------------------------------------------
# STEP B.5: SHUFFLE DATA
# -------------------------------------------------

from sklearn.utils import shuffle

X, y_encoded, y_onehot = shuffle(
    X,
    y_encoded,
    y_onehot,
    random_state=42
)

print("\nSTEP B.5 DONE")
print("Data shuffled")

# -------------------------------------------------
# STEP C: TRAIN / VALIDATION SPLIT
# -------------------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y_onehot,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nSTEP C DONE")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("y_train:", y_train.shape)
print("y_val:", y_val.shape)

# -------------------------------------------------
# STEP D: SAVE DATA
# -------------------------------------------------

SAVE_DIR = "data7/words_grutrn"
os.makedirs(SAVE_DIR, exist_ok=True)

np.save(os.path.join(SAVE_DIR, "X_train.npy"), X_train)
np.save(os.path.join(SAVE_DIR, "X_val.npy"), X_val)
np.save(os.path.join(SAVE_DIR, "y_train.npy"), y_train)
np.save(os.path.join(SAVE_DIR, "y_val.npy"), y_val)

# label encoder
with open(os.path.join(SAVE_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)

# class names
np.save(
    os.path.join(SAVE_DIR, "classes.npy"),
    label_encoder.classes_
)

print("\nSTEP D DONE")
print("GRU-ready data saved successfully.")