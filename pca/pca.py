import numpy as np
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None  # This will store the principal components
        self.mean = None  # Mean of the dataset for centering

    def fit(self, X):
        # Center the data (subtract the mean)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        covariance_matrix = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # sort the eigenvectors by decreasing eigenvalues
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # plt.figure(figsize=(12, 8))
        # bars = plt.bar(range(len(sorted_eigenvalues)), sorted_eigenvalues)

        # # Adding labels above the bars
        # for i, bar in enumerate(bars):
        #     plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
        #             f'{sorted_indices[i]}', ha='center', va='bottom', fontsize=6)

        # # Adding labels and title
        # plt.xlabel('Eigenvalue Index (Sorted)')
        # plt.ylabel('Eigenvalue Magnitude')
        # plt.title('Eigenvalue Bar Plot with Component Labels')
        # plt.show()

        # Select the top n_components eigenvectors (principal components)
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        # Project the data onto the principal components
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def inverse_transform(self, X_reduced):                           #Xreducecd = X . components
                                                                      #X = Xreduced.(components)^T    (transpose since components is orthonormal vector)
        # ceconstruct the original data from the reduced data
        return np.dot(X_reduced, self.components.T) + self.mean 
    
    def reconstruction_error(self, X):
        # Calculate reconstruction error between original and reconstructed data
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)
        error = np.mean((X - X_reconstructed) ** 2)
        return error
    def checkPCA(self, X):
        # Verify that the dimensionality is reduced as expected
        error = self.reconstruction_error(X)
        if(error < 0.5):
            print(f"Error in reconstruction is:{error}")
            return True
        else:
            return False
