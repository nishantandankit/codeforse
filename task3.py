import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# --- 1. Load and Preprocess the MNIST Data ---
# Load the MNIST dataset, a large database of handwritten digits. [1, 2, 5, 6, 7]
(X_train_full, y_train_full), (X_test_full, y_test_full) = mnist.load_data()

# Use a subset of the data to speed up training for this example
X_train, y_train = X_train_full[:10000], y_train_full[:10000]
X_test, y_test = X_test_full[:2000], y_test_full[:2000]

# Normalize the pixel values from integers (0-255) to floats (0-1)
# This is a common and important step for training neural networks.
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape data for CNN input (samples, height, width, channels)
# MNIST images are grayscale, so they have one channel.
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print(f"--- Data Loaded and Preprocessed ---")
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Visualize an example image to verify
plt.imshow(X_train[0].reshape(28, 28), cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()


# --- 2. Build the CNN Model ---
# Using the Keras Sequential API for building the model layer by layer. [3]
model = Sequential([
    # First convolutional layer: 32 filters, 3x3 kernel size. 'relu' activation is common.
    # The input shape must be specified for the first layer. [21]
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),

    # Max pooling layer to reduce the spatial dimensions (down-sampling). [21]
    MaxPooling2D((2, 2)),

    # Second convolutional layer to learn more complex features.
    Conv2D(64, (3, 3), activation='relu'),

    # Second max pooling layer.
    MaxPooling2D((2, 2)),

    # Flatten the 2D feature maps into a 1D vector before feeding to the dense layers. [21]
    Flatten(),

    # A dense (fully connected) layer with 128 units.
    Dense(128, activation='relu'),

    # The final output layer. It has 10 units (one for each digit 0-9).
    # 'softmax' is used for multi-class classification to output a probability distribution. [3, 21]
    Dense(10, activation='softmax')
])

model.summary()

# --- 3. Compile the Model ---
# Configure the model for training.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use this loss function for integer labels.
              metrics=['accuracy'])

# --- 4. Train the Model ---
print("\n--- Training the CNN Model ---")
# We train the model on our training data and validate it on our test data.
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# --- 5. Performance Evaluation ---
# Evaluate the final performance of the model on the unseen test dataset.
print("\n--- Evaluating Model Performance on Test Data ---")
loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# --- 6. Visualize Training History ---
# Plotting the training and validation accuracy helps to check for overfitting.
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plotting the training and validation loss.
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
