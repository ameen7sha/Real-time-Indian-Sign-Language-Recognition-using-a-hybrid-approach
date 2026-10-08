import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load model
model = load_model("cnn_")

# Load your test data
X_test = np.load("data/alphabets_cnn/X_train.npy")
y_test = np.load("data/alphabets_cnn/y_train.npy")

# Predict
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("CNN Confusion Matrix")
plt.savefig("cnn_confusion_matrix.png")
plt.show()