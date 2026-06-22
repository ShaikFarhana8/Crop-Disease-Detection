import os
import matplotlib.pyplot as plt
from PIL import Image

dataset_path = "plantvillage dataset/color"

classes = os.listdir(dataset_path)

plt.figure(figsize=(20, 20))

for i, cls in enumerate(classes[:16]):  # first 16 classes
    class_path = os.path.join(dataset_path, cls)

    image_name = os.listdir(class_path)[0]

    image_path = os.path.join(class_path, image_name)

    img = Image.open(image_path)

    plt.subplot(4, 4, i + 1)
    plt.imshow(img)
    plt.title(cls[:20])
    plt.axis("off")

plt.tight_layout()
plt.savefig("sample_images.png")
plt.show()