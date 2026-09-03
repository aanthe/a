!pip install -q diffusers transformers accelerate torch

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x, _), (_, _) = tf.keras.datasets.mnist.load_data()
x = x[:10000].astype("float32") / 255.0
x = x[..., None]

noise = tf.random.normal(tf.shape(x))
noisy_x = x + 0.5 * noise

model = tf.keras.Sequential([
    tf.keras.layers.Input((28, 28, 1)),
    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    tf.keras.layers.Conv2D(1, 3, padding="same")
])

model.compile(optimizer="adam", loss="mse")

early_stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)

model.fit(noisy_x, noise, epochs=50, batch_size=128, callbacks=[early_stop])

img = tf.random.normal((1, 28, 28, 1))

for _ in range(100):
    img -= 0.02 * model(img, training=False)

img = img[0, :, :, 0]

plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("Generated MNIST Image")
plt.show()
