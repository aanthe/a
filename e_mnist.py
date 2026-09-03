"""
Purpose:
This file shows a simple image denoising and generation-style experiment using MNIST digits.
We want to teach a model to remove noise from digit images, then repeatedly apply that idea to random noise.

What we are trying to achieve:
Start with noisy or random pixel data and push it toward something that looks more like a handwritten MNIST digit.

Overall flow:
1. Load and scale MNIST images.
2. Add random noise to create a denoising task.
3. Train a small convolutional model to predict noise.
4. Start from pure random noise.
5. Repeatedly subtract predicted noise to get a cleaner image.
6. Display the final image.
"""

Theory:
MNIST is a standard dataset of handwritten digits.
This script trains a small convolutional network to predict noise, then repeatedly removes noise from a random image so it becomes more digit-like.

Viva:
Q: What is MNIST?
A: A dataset of handwritten digits from 0 to 9.
Q: Why add noise to images?
A: To create a denoising learning task.
Q: Why use convolution layers?
A: They learn local image patterns like edges and strokes.
Q: Why start from random noise?
A: To test whether repeated denoising can create a structured image.

Output:
The model trains on noisy MNIST images and then shows one generated grayscale image that may look loosely like a handwritten digit.

# Run this cell once in Google Colab.
# Why: these packages are needed if you are exploring diffusion-style generation workflows.
!pip install -q diffusers transformers accelerate torch

import tensorflow as tf  # TensorFlow provides the dataset loader and neural network tools.
import numpy as np  # NumPy is available for array handling if needed.
import matplotlib.pyplot as plt  # Matplotlib is used to display the final image.

# Term: MNIST means a famous dataset of handwritten digits from 0 to 9.
# Load MNIST.
# Why: MNIST gives us simple handwritten digits, which are a classic starting point for image learning.
(x, _), (_, _) = tf.keras.datasets.mnist.load_data()
x = x[:10000].astype("float32") / 255.0  # Use only 10,000 images and scale pixels from 0-255 down to 0-1.
x = x[..., None]  # Add a channel dimension so the data shape becomes (28, 28, 1).

# Term: noise means random unwanted variation added to data.
# Add noise.
# Why: the model will learn to remove noise by seeing noisy images and the noise itself.
noise = tf.random.normal(tf.shape(x))  # Create random noise with the same shape as the images.
noisy_x = x + 0.5 * noise  # Mix the original image with noise to make the learning task harder.

# Term: convolution means a sliding-window operation that learns local patterns like edges and strokes.
# Term: neural network means a layered model that learns patterns from data.
# Denoising model.
# Why: a convolutional neural network can learn local image patterns and predict the noise.
model = tf.keras.Sequential([
    tf.keras.layers.Input((28, 28, 1)),  # Input shape matches a single MNIST image.
    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),  # Learn 32 simple features like edges and blobs.
    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),  # Learn richer features from the first layer's output.
    tf.keras.layers.Conv2D(1, 3, padding="same")  # Predict one noise value for each pixel.
])

# Term: optimizer means the algorithm that updates model weights to reduce error.
# Term: loss means a number that measures how wrong the model is.
model.compile(optimizer="adam", loss="mse")  # Adam adjusts weights efficiently; MSE measures squared prediction error.

# Term: early stopping means ending training when the model stops improving.
# Early stopping.
# Why: stop training once improvement stalls, which saves time and avoids overfitting.
early_stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)

# Term: epoch means one full pass through the training data.
# Term: batch means a small group of samples processed together.
# Train.
# Why: the model learns by comparing its noise prediction with the true added noise.
model.fit(noisy_x, noise, epochs=50, batch_size=128, callbacks=[early_stop])

# Term: iterative means repeating a process many times.
# Start from random noise.
# Why: generation begins from chaos, then repeated denoising moves it toward a structured image.
img = tf.random.normal((1, 28, 28, 1))  # Create one random image-shaped tensor.

# Iterative denoising.
# Why: each step subtracts the model's estimate of noise, gradually making the image cleaner.
for _ in range(100):
    img -= 0.02 * model(img, training=False)  # Move slightly against the predicted noise.

# Display.
# Why: convert the tensor into a plain 2D image so matplotlib can show it.
img = img[0, :, :, 0]  # Remove batch and channel dimensions.

plt.imshow(img, cmap="gray")  # Show grayscale pixels.
plt.axis("off")  # Hide axis numbers because they are not useful for a generated image.
plt.title("Generated MNIST Image")  # Explain what the image is meant to represent.
plt.show()  # Render the final result.
