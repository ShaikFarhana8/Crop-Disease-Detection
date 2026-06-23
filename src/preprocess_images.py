import os
from PIL import Image

dataset_path = "plantvillage dataset/color"
target_size = (224, 224)

count = 0

for cls in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, cls)

    if os.path.isdir(class_path):
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)

            try:
                img = Image.open(image_path)
                img = img.resize(target_size)

                count += 1

                if count % 1000 == 0:
                    print(f"Processed {count} images...")

            except Exception:
                pass

print(f"\nSuccessfully processed {count} images")