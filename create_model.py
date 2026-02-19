import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Create the directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Define a simple CNN (Module 3 architecture)
model = models.Sequential([
    layers.Input(shape=(160, 160, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(4, activation='softmax') # 4 classes: Real, Photo, Video, 3D
])

# Save it properly
model.save('models/antispoof_model.h5')
print("Model created successfully in models/antispoof_model.h5")