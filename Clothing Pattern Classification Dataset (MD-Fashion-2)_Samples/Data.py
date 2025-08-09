import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
        class_name = img_name.split("_")[0]  # Extract class prefix from filename
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

print(f"[DONE] Loaded {len(X)} images across {len(label_map)} classes: {list(label_map.keys())}")
