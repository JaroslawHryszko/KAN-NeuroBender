import numpy as np

def phi(x):
    """ A simple nonlinear function used in the network. """
    return np.sin(x)

def kolmogorov_arnold_network(x, V, W):
    """
    Kolmogorov-Arnold representation of a function.
    
    :param x: Input vector (numpy array).
    :param V: Matrix of weights for combining inputs (numpy array).
    :param W: Weights for combining single-variable functions (numpy array).
    :return: Output of the network (float).
    """
    # Apply V to inputs to generate terms for phi
    z = np.dot(V, x)
    
    # Apply the nonlinear function phi to each component
    phi_z = phi(z)
    
    # Sum up all the transformed components using weights W
    output = np.dot(W, phi_z)
    
    return output

# Example input
x = np.array([0.5, -1.2, 0.3])

# Random weights for demonstration (usually needs careful initialization)
V = np.random.rand(3, 3)  # 3 inputs, 3 transformed features
W = np.random.rand(3)     # Weights for combining the outputs of phi

# Compute the output
output = kolmogorov_arnold_network(x, V, W)
print("Output of the network:", output)
