import os
import numpy as np
from PIL import Image

dataset_path = "plantvillage dataset/color"

first_class = os.listdir(dataset_path)[0]

class_path = os.path.join(dataset_path, first_class)

first_image = os.listdir(class_path)[0]

image_path = os.path.join(class_path, first_image)

img = Image.open(image_path)

img = img.resize((224, 224))

img_array = np.array(img)

print("Before Normalization")
print("Min Pixel:", img_array.min())
print("Max Pixel:", img_array.max())

normalized_img = img_array / 255.0

print("\nAfter Normalization")
print("Min Pixel:", normalized_img.min())
print("Max Pixel:", normalized_img.max())