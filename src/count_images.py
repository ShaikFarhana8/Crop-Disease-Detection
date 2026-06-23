import os

dataset_path = "plantvillage dataset/color"

total_images = 0

for cls in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, cls)

    if os.path.isdir(class_path):
        count = len(os.listdir(class_path))
        total_images += count

print("Total Images:", total_images)