from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load model and label map
model = tf.keras.models.load_model("fabric_pattern_model.h5")
with open("label_map.json", "r") as f:
    label_map = json.load(f)

# Create ordered list of categories
categories = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]

app = Flask(__name__)

def preprocess_image(img):
    img = img.convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 128, 128, 3)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '':
            return "No valid image uploaded", 400
        try:
            img = Image.open(file.stream)
            img_array = preprocess_image(img)
            prediction = model.predict(img_array)
            predicted_class = categories[np.argmax(prediction)]
            confidence = round(100 * np.max(prediction), 2)

            # Debug
            print("Probabilities:", prediction[0])
            for i, prob in enumerate(prediction[0]):
                print(f"{categories[i]}: {prob:.2f}")

            return render_template('result.html', prediction=predicted_class, confidence=confidence)
        except Exception as e:
            return f"Error processing image: {e}", 500

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
