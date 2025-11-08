import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

data_dir = "./Dataset"
img_size = 64

images = []
labels = []

# Folder-label map
label_map = {"01": 0, "02": 1, "03": 2}

for folder in label_map:
    folder_path = os.path.join(data_dir, folder)
    if not os.path.isdir(folder_path):
        print("⚠ Folder not found:", folder_path)
        continue

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print("⚠ Skipping invalid:", img_path)
                continue

            img = cv2.resize(img, (img_size, img_size))
            images.append(img)
            labels.append(label_map[folder])

images = np.array(images).reshape(-1, img_size, img_size, 1) / 255.0
labels = to_categorical(np.array(labels), num_classes=3)

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(img_size, img_size, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(3, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

model.save("waste_cnn_model.h5")
print("✅ Model Saved: waste_cnn_model.h5")
