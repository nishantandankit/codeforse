import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

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

    
    images = np.array(images).reshape(-1, 10, 10, 1)
    labels = np.array(labels)
    return images, labels

X_train, y_train = generate_image_data(100) 
X_test, y_test = generate_image_data(20)  

print(f"--- Data Generated ---")
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")


plt.subplot(1, 2, 1)
plt.imshow(X_train[0].reshape(10, 10), cmap='gray')
plt.title("Class 0: Horizontal Line")
plt.subplot(1, 2, 2)
plt.imshow(X_train[1].reshape(10, 10), cmap='gray')
plt.title("Class 1: Vertical Line")
plt.show()

model = Sequential([
    Conv2D(8, (3, 3), activation='relu', input_shape=(10, 10, 1)),


    MaxPooling2D((2, 2)),

  
    Flatten(),

 
    Dense(10, activation='relu'),

   
    Dense(1, activation='sigmoid')
])

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("\n--- Training the CNN Model ---")
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

print("\n--- Evaluating Model Performance on Test Data ---")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
