import unittest
import numpy as np
from tensorflow.keras.models import Model
from integrated_model import IntegratedModel, load_preprocess_data
from wae_mnist import build_encoder, build_decoder

class TestIntegratedModel(unittest.TestCase):
    def setUp(self):
        # Load data
        self.x_train, self.x_test = load_preprocess_data()

        # Initialize models
        self.encoder = build_encoder()
        self.decoder = build_decoder()
        self.V = np.random.rand(784, 784)  # Example weight initialization
        self.W = np.random.rand(784)

        # Create instance of the integrated model
        self.model = IntegratedModel(self.encoder, self.decoder, self.V, self.W)

    def test_preprocess_with_ka_network_shape(self):
        """Test preprocessing maintains expected shape."""
        processed = self.model.preprocess_with_ka_network(self.x_test[:10])
        self.assertEqual(processed.shape, self.x_test[:10].shape)

    def test_predict_output_shape(self):
        """Test that the predict method returns output with the correct shape."""
        reconstructed = self.model.predict(self.x_test[:10])
        self.assertEqual(reconstructed.shape, self.x_test[:10].shape)

    def test_integration_flow(self):
        """Test that data flows through the full model without errors and maintains shape."""
        original_shape = self.x_test[:5].shape
        reconstructed = self.model.predict(self.x_test[:5])
        self.assertEqual(reconstructed.shape, original_shape)

if __name__ == '__main__':
    unittest.main()
