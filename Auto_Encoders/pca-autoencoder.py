#PCA

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class PcaAutoencoder:
    def __init__(self, n_components=None):
        """
        Initialize PCA Autoencoder

        Args:
            n_components (int): Number of principal components to keep
        """
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None

    def fit(self, X):
        """
        Fit the PCA model by computing eigenvalues and eigenvectors

        Args:
            X (np.array): Input data of shape (n_samples, n_features)
        """
        # Center the data
        X = X.cpu().numpy()
        X = X.reshape(X.shape[0], -1)

        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Compute covariance matrix
        n_samples = X_centered.shape[0]
        cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store components and variance information
        if self.n_components is None:
            self.n_components = min(X.shape)

        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ = eigenvalues[:self.n_components]
        total_var = eigenvalues.sum()
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / total_var
        self.singular_values_ = np.sqrt(eigenvalues[:self.n_components] * (n_samples - 1))

        return self

    def encode(self, X):
        """
        Transform data to reduced dimensional space

        Args:
            X (np.array): Input data of shape (n_samples, n_features)

        Returns:
            np.array: Transformed data in reduced dimensional space
        """
        X = X.reshape(X.shape[0], -1) # Reshape to 2D
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)

    def forward(self, X):
        """
        Transform data to reduced dimensional space and back to original space

        Args:
            X (np.array): Input data of shape (n_samples, n_features)

        Returns:
            np.array: Reconstructed data in original space
        """
        latent = self.encode(X)
        reconstructed = np.dot(latent, self.components_.T) + self.mean_
        return reconstructed

def find_optimal_components(X_train, X_val, max_components=100, step=5):
    """
    Find optimal number of components using elbow method

    Args:
        X_train (np.array): Training data
        X_val (np.array): Validation data
        max_components (int): Maximum number of components to try
        step (int): Step size for number of components

    Returns:
        tuple: Optimal number of components and list of reconstruction errors
    """
    X_val = X_val.cpu().numpy()
    X_val = X_val.reshape(X_val.shape[0],-1)
    n_components_range = range(step, max_components + step, step)
    reconstruction_errors = []

    for n in n_components_range:
        # Train PCA with n components
        pca = PcaAutoencoder(n_components=n)
        pca.fit(X_train)

        # Get reconstruction error on validation set
        reconstructed = pca.forward(X_val)
        mse = np.mean((X_val - reconstructed) ** 2)
        reconstruction_errors.append(mse)

        print(f"Components: {n}, MSE: {mse:.6f}")

    # Find elbow point using second derivative method
    errors = np.array(reconstruction_errors)
    diffs = np.diff(errors, 2)  # Second derivative
    elbow_idx = np.argmax(diffs) + 2  # Add 2 due to diff operation
    optimal_components = n_components_range[elbow_idx]

    return optimal_components, n_components_range, reconstruction_errors

def plot_elbow_curve(n_components_range, reconstruction_errors, optimal_components):
    """Plot elbow curve with optimal point marked"""
    plt.figure(figsize=(10, 6))
    plt.plot(n_components_range, reconstruction_errors, 'b-', label='Reconstruction Error')
    plt.plot(optimal_components, reconstruction_errors[optimal_components//5 - 1], 'ro',
             label=f'Optimal Components (k={optimal_components})')
    plt.xlabel('Number of Components')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.title('Elbow Plot: Reconstruction Error vs Number of Components')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_reconstructions(original_images, reconstructed_images, num_images=10):
    """
    Visualize original and reconstructed images side by side

    Args:
        original_images (np.array): Original images
        reconstructed_images (np.array): Reconstructed images
        num_images (int): Number of images to display
    """
    plt.figure(figsize=(20, 4))
    for i in range(num_images):
        # Original image
        plt.subplot(2, num_images, i + 1)
        plt.imshow(original_images[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Original Images')

        # Reconstructed image
        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(reconstructed_images[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Reconstructed Images')

    plt.tight_layout()
    plt.show()


# Train final model with optimal components
pca_autoencoder = PcaAutoencoder(n_components='optimal_components')
pca_autoencoder.fit(X_train)

# Get reconstructions for test set
reconstructed_test = pca_autoencoder.forward(X_test)

# Visualize results
visualize_reconstructions(X_test, reconstructed_test, num_images=10)

# Print final test reconstruction error
test_mse = np.mean((X_test.cpu().numpy().reshape(X_test.shape[0],-1) - reconstructed_test) ** 2)
print(f'Final test reconstruction MSE with {optimal_components} components: {test_mse:.6f}')
print(f'Explained variance ratio: {sum(pca_autoencoder.explained_variance_ratio_):.4f}')
