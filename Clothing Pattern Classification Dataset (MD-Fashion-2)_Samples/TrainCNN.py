import os
import cv2
import numpy as np
import json
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load and preprocess data
img_size = 128
data_dir = r"C:\Users\nandi\OneDrive\Desktop\Fabric\Clothing Pattern Classification Dataset (MD-Fashion-2)_Samples"

X, y = [], []
label_map = {}
label_counter = 0
valid_exts = ['.jpg', '.jpeg', '.png']

print("[INFO] Scanning image files...")
for img_name in os.listdir(data_dir):
    if not any(img_name.lower().endswith(ext) for ext in valid_exts):
        continue
    img_path = os.path.join(data_dir, img_name)
    try:
        class_name = img_name.split("_")[0]
        if class_name not in label_map:
            label_map[class_name] = label_counter
            label_counter += 1
        label = label_map[class_name]

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Could not read image: {img_path}")
            continue
        img = cv2.resize(img, (img_size, img_size))
        X.append(img)
        y.append(label)
    except Exception as e:
        print(f"[ERROR] {img_path} → {e}")

if not y:
    raise ValueError("No images were loaded. Please check dataset contents and image formats.")

X = np.array(X) / 255.0
y = to_categorical(np.array(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance
class_indices = np.argmax(y, axis=1)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(class_indices), y=class_indices)
class_weight_dict = dict(enumerate(class_weights))

print(f"[INFO] Loaded {len(X)} images across {len(label_map)} classes: {label_map}")
print(f"[INFO] Class distribution: {Counter(class_indices)}")

# Build and train model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_map), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("[INFO] Starting training...")
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), class_weight=class_weight_dict)

# Save model and label map
model.save("fabric_pattern_model.h5")
with open("label_map.json", "w") as f:
    json.dump(label_map, f)
print("[INFO] Training complete and model saved.")
