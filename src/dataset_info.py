import os

dataset_path = "plantvillage dataset/color"

classes = sorted(os.listdir(dataset_path))

print("=" * 50)
print("PLANTVILLAGE DATASET REPORT")
print("=" * 50)

print(f"\nNumber of Classes: {len(classes)}\n")

total_images = 0

for cls in classes:
    class_path = os.path.join(dataset_path, cls)

    if os.path.isdir(class_path):
        image_count = len(os.listdir(class_path))
        total_images += image_count

        print(f"{cls}: {image_count} images")

print("\n" + "=" * 50)
print(f"Total Images: {total_images}")
print("=" * 50)