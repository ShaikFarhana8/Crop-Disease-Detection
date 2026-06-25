import tensorflow as tf
from tensorflow.keras import layers, models

# Create CNN Model
model = models.Sequential([

    # Input Layer + Normalization
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),

    # First Convolution Block
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Second Convolution Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Third Convolution Block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Flatten Layer
    layers.Flatten(),

    # Fully Connected Layer
    layers.Dense(128, activation='relu'),

    # Dropout Layer
    layers.Dropout(0.5),

    # Output Layer
    layers.Dense(38, activation='softmax')
])

# Build Model
model.build((None, 224, 224, 3))

# Print Information
print("=" * 50)
print("CNN MODEL CREATED SUCCESSFULLY")
print("=" * 50)

print("\nInput Shape: (224, 224, 3)")
print("Output Classes: 38")

print("\nModel Parameters:")
print(model.count_params())

print("\nDay 8 Completed Successfully!")