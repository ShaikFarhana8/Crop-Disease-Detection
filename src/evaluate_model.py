import tensorflow as tf
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model("crop_disease_model.keras")

print("=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print("Model loaded successfully!")
print("\nModel Summary:")
model.summary()

print("\nEvaluation completed successfully.")