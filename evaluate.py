# evaluate.py

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Path to saved model
model_path = '/content/drive/MyDrive/detector_emotion/emotion_model.keras'

# Path to test data
test_dir = '/content/dataset/test'

# Emotion labels (in order)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load model
model = load_model(model_path)

# Load test data
X_test = []
y_test = []

for label_idx, emotion in enumerate(emotion_labels):
    folder = os.path.join(test_dir, emotion)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (48, 48))
            X_test.append(img)
            y_test.append(label_idx)

# Convert to numpy arrays
X_test = np.array(X_test).reshape(-1, 48, 48, 1) / 255.0
y_test = to_categorical(np.array(y_test), num_classes=7)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")
print(f"📉 Test Loss: {loss:.4f}")
