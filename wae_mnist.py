import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

# Load MNIST data
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype('float32') / 255.0
x_test = np.expand_dims(x_test, -1).astype('float32') / 255.0

# Encoder architecture
def build_encoder():
    inputs = Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu', strides=2, padding='same')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)
    z_mean = layers.Dense(10)(x)
    z_log_var = layers.Dense(10)(x)
    encoder = Model(inputs, [z_mean, z_log_var], name="encoder")
    return encoder

# Decoder architecture
def build_decoder():
    latent_inputs = Input(shape=(10,))
    x = layers.Dense(7 * 7 * 64, activation='relu')(latent_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=2, padding='same')(x)
    outputs = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    decoder = Model(latent_inputs, outputs, name="decoder")
    return decoder

encoder = build_encoder()
decoder = build_decoder()

# Sampling layer
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Wasserstein loss
def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# VAE model
class WAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(WAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = Sampling()([z_mean, z_log_var])
        reconstructed = self.decoder(z)
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        return reconstructed

# Instantiate and compile WAE
wae = WAE(encoder, decoder)
wae.compile(optimizer='adam', loss=wasserstein_loss)

# Train the model
wae.fit(x_train, x_train, epochs=10, batch_size=32, validation_data=(x_test, x_test))

# Display original and reconstructed images
def plot_images(original, reconstructed):
    n = 10  # Number of digits to display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

# Predict on test data
test_images = x_test[:10]
reconstructed_images = wae.predict(test_images)
plot_images(test_images, reconstructed_images)
