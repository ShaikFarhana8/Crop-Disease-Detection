import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =====================================
# Load Trained Model
# =====================================
model = tf.keras.models.load_model("crop_disease_model.keras")

# =====================================
# Load Class Names
# =====================================
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# =====================================
# Test Image Path
# =====================================
image_path = "test_images/apple_scab.JPG"   # Change to your image

# =====================================
# Load Image
# =====================================
img = image.load_img(image_path, target_size=(224, 224))

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# MobileNetV2 preprocessing
img_array = preprocess_input(img_array)

# =====================================
# Predict
# =====================================
prediction = model.predict(img_array, verbose=0)[0]

predicted_index = np.argmax(prediction)
confidence = prediction[predicted_index]

# =====================================
# Print Result
# =====================================
print("=" * 60)
print("Crop Disease Prediction")
print("=" * 60)
print("Image            :", image_path)
print("Predicted Class  :", class_names[predicted_index])
print(f"Confidence       : {confidence * 100:.2f}%")

print("\nTop 5 Predictions")
print("-" * 60)

top5 = np.argsort(prediction)[::-1][:5]

for idx in top5:
    print(f"{class_names[idx]:45} {prediction[idx] * 100:.2f}%")

print("=" * 60)