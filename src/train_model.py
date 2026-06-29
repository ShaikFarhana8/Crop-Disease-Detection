import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import json

# Dataset path
dataset_path = "plantvillage dataset/color"

# Load Training Dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(224, 224),
    batch_size=32
)

# Load Validation Dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(224, 224),
    batch_size=32
)

# Print Class Information
print("\n===================================")
print("Number of Classes:", len(train_ds.class_names))
print("===================================")

for i, class_name in enumerate(train_ds.class_names):
    print(f"{i}: {class_name}")

# Save Class Names
with open("class_names.json", "w") as f:
    json.dump(train_ds.class_names, f)

print("\nClass names saved as class_names.json")

# Build CNN Model
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Rescaling(1./255),

    layers.Conv2D(32, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    # Automatically matches the dataset classes
    layers.Dense(len(train_ds.class_names), activation="softmax")
])

# Compile Model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining Started...\n")

# Train Model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

# Accuracy Plot
plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_plot.png")
plt.show()

# Loss Plot
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_plot.png")
plt.show()

# Save Model
model.save("crop_disease_model.keras")

print("\n===================================")
print("Model Training Completed Successfully!")
print("Model saved as crop_disease_model.keras")
print("Class names saved as class_names.json")
print("===================================")