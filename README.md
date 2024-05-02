# NeuroBender: Enhanced Image Processing with K-A Networks & WAE

Welcome to NeuroBender, where the mystical powers of mathematical complexity meet the robust stability of autoencoders! This project combines the Kolmogorov-Arnold network with a Wasserstein Auto-Encoder to provide a unique approach to image processing on the MNIST dataset. Dive into a world where digits are not just recognized but transformed and reconstructed in new and interesting ways!

## Project Overview

NeuroBender takes the classical MNIST dataset and applies a two-stage process:
1. **Transformation using the Kolmogorov-Arnold Network**: This step involves a complex transformation of the image data to enhance features.
2. **Reconstruction using the Wasserstein Auto-Encoder**: Post-transformation, the data is fed into a WAE to reduce dimensionality and reconstruct the images.

This approach is designed to test the synergy between complex mathematical transformations and modern deep learning techniques.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You'll need Python 3.8 or later, and the following packages:
- TensorFlow
- NumPy
- Matplotlib

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/JaroslawHryszko/neurobender.git
   ```

2. **Navigate to the project directory:**
   ```bash
   cd neurobender
   ```

3. **Install required Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

To run NeuroBender and see it in action:
```bash
python integrated_model.py
```

This will execute the process on a subset of MNIST data and display the original and reconstructed images.

## Code Structure

- `kolmogorov_arnold_network.py`: Contains the implementation of the Kolmogorov-Arnold network.
- `wae_mnist.py`: Houses the Wasserstein Auto-Encoder setup.
- `integrated_model.py`: Integrates both models and includes preprocessing and prediction logic.
- `test_integrated_model.py`: Contains unit tests for verifying the functionality of integrated systems.
