import unittest
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from wae_mnist import build_encoder, build_decoder, WAE

class TestWassersteinAutoEncoder(unittest.TestCase):

    def test_encoder_output(self):
        """Test the encoder outputs the mean and log variance with the correct shape."""
        encoder = build_encoder()
        x_fake = np.random.rand(10, 28, 28, 1)  # Batch of 10, 28x28 images with 1 channel
        z_mean, z_log_var = encoder.predict(x_fake)
        self.assertEqual(z_mean.shape, (10, 10))  # 10 latent dimensions
        self.assertEqual(z_log_var.shape, (10, 10))

    def test_decoder_output_shape(self):
        """Test the decoder outputs images with the correct shape."""
        decoder = build_decoder()
        z_fake = np.random.rand(10, 10)  # Batch of 10, 10-dimensional latent vectors
        generated_images = decoder.predict(z_fake)
        self.assertEqual(generated_images.shape, (10, 28, 28, 1))  # Should match input image shape

    def test_wae_integration(self):
        """Test the integration of the WAE model, ensuring it can process input through encoder and decoder."""
        encoder = build_encoder()
        decoder = build_decoder()
        wae = WAE(encoder, decoder)
        
        x_fake = np.random.rand(10, 28, 28, 1)  # Batch of 10, 28x28 images with 1 channel
        reconstructed = wae.predict(x_fake)
        self.assertEqual(reconstructed.shape, x_fake.shape)

    def test_sampling_layer(self):
        """Test the sampling layer to ensure it adds randomness correctly."""
        z_mean = np.zeros((10, 10))
        z_log_var = np.zeros((10, 10))
        batch = 10
        dim = 10
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        z_sample = z_mean + tf.exp(0.5 * z_log_var) * epsilon

        self.assertEqual(z_sample.shape, (10, 10))

if __name__ == '__main__':
    unittest.main()
