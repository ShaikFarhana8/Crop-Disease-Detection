import os
import pandas as pd
import matplotlib.pyplot as plt

dataset_path = "plantvillage dataset/color"

classes = []
counts = []

for cls in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, cls)

    if os.path.isdir(class_path):
        image_count = len(os.listdir(class_path))

        classes.append(cls)
        counts.append(image_count)

df = pd.DataFrame({
    "Class": classes,
    "Images": counts
})

print(df)

plt.figure(figsize=(16,8))
plt.bar(classes, counts)

plt.xticks(rotation=90)
plt.xlabel("Disease Classes")
plt.ylabel("Number of Images")
plt.title("PlantVillage Dataset Class Distribution")

plt.tight_layout()

plt.savefig("class_distribution.png")

print("\nChart saved successfully!")

plt.show()