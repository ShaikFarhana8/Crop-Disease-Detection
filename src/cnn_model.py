import tensorflow as tf
from tensorflow.keras import layers, models

# Create CNN Model
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Rescaling(1./255),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(38, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("=" * 50)
print("CNN MODEL COMPILED SUCCESSFULLY")
print("=" * 50)

print(f"Optimizer      : Adam")
print(f"Loss Function  : Sparse Categorical Crossentropy")
print(f"Metric         : Accuracy")
print(f"Total Parameters: {model.count_params()}")