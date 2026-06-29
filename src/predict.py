import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing import image

# Load Trained Model
model = tf.keras.models.load_model("crop_disease_model.keras")

# Load Class Names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Test Image Path
image_path = "test_images/apple_scab1.JPG"   # Change this to your test image

# Load and Preprocess Image
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict
prediction = model.predict(img_array, verbose=0)

predicted_index = np.argmax(prediction)
confidence = np.max(prediction)

# Display Result
print("=" * 50)
print("Crop Disease Prediction")
print("=" * 50)
print("Image:", image_path)
print("Predicted Class :", class_names[predicted_index])
print(f"Confidence : {confidence * 100:.2f}%")
print("=" * 50)