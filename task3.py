import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

# --- 1. Generate Synthetic Image Data ---
# Since no data is provided, we create our own simple images.
# Class 0: Horizontal Line
# Class 1: Vertical Line
def generate_image_data(num_samples):
    images = []
    labels = []
    for i in range(num_samples):
        image = np.zeros((10, 10))
        if i % 2 == 0:
            # Class 0: Horizontal line
            image[5, :] = 1.0  # Draw a line in the middle row
            labels.append(0)
        else:
            # Class 1: Vertical line
            image[:, 5] = 1.0  # Draw a line in the middle column
            labels.append(1)
        # Add some noise to make it slightly more challenging
        image += np.random.normal(0, 0.1, image.shape)
        images.append(image)

    # Reshape for CNN input (samples, height, width, channels)
    images = np.array(images).reshape(-1, 10, 10, 1)
    labels = np.array(labels)
    return images, labels

# Generate training and testing data
X_train, y_train = generate_image_data(100) # 100 images for training
X_test, y_test = generate_image_data(20)   # 20 images for testing

print(f"--- Data Generated ---")
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Visualize an example image from each class
plt.subplot(1, 2, 1)
plt.imshow(X_train[0].reshape(10, 10), cmap='gray')
plt.title("Class 0: Horizontal Line")
plt.subplot(1, 2, 2)
plt.imshow(X_train[1].reshape(10, 10), cmap='gray')
plt.title("Class 1: Vertical Line")
plt.show()


# --- 2. Build the CNN Model ---
# Using the Keras Sequential API, which is great for beginners.
model = Sequential([
    # Convolutional layer to find features (lines)
    # 8 filters, 3x3 kernel size, 'relu' activation, input shape
    Conv2D(8, (3, 3), activation='relu', input_shape=(10, 10, 1)),

    # Pooling layer to reduce dimensionality
    MaxPooling2D((2, 2)),

    # Flatten the 2D features into a 1D vector
    Flatten(),

    # A dense layer for classification
    Dense(10, activation='relu'),

    # Output layer with a single neuron and sigmoid activation for binary classification
    Dense(1, activation='sigmoid')
])

model.summary()

# --- 3. Compile the Model ---
# Configure the model for training
model.compile(optimizer='adam',
              loss='binary_crossentropy', # Good for two-class problems
              metrics=['accuracy'])

# --- 4. Train the Model ---
print("\n--- Training the CNN Model ---")
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# --- 5. Performance Evaluation ---
# Evaluate the model on the test dataset.
print("\n--- Evaluating Model Performance on Test Data ---")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
