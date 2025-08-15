import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, n_components, max_iter=100, tol=1e-3):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol        
        self.means = None
        self.covariances = None
        
        self.weights = None

    def fit(self, X):
        """
        Fit the GMM to the dataset using the Expectation-Maximization (EM) algorithm.
        param X: Input data of shape (n_samples, n_features)
        """
        n_samples, n_features = X.shape
        
        # randomly select n_components points from X to initialize the means
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        
        # initialize covariance matrices to identity matrices for each component
        self.covariances = np.array([np.eye(n_features)] * self.n_components)
        
        # initialize the weights equally (each component has equal responsibility initially)
        self.weights = np.ones(self.n_components) / self.n_components
        
        # list to store the log-likelihood values at each iteration for convergence checking
        log_likelihoods = []
        
        # Perform the EM algorithm for a maximum of self.max_iter iterations
        for iteration in range(self.max_iter):
            #E-step: Calculate responsibilities (probabilities that each sample belongs to each component)
            responsibilities = self._e_step(X)
            
            # M-step: Update the model parameters (means, covariances, and weights)
            self._m_step(X, responsibilities)
            
            # Compute the log-likelihood of the current model
            log_likelihood = self.getLikelihood(X)
            
            log_likelihoods.append(log_likelihood)
            
            # check if the change in log-likelihood is below the tolerance threshold (convergence)
            if iteration > 0 and np.abs(log_likelihood - log_likelihoods[-2]) < self.tol:
                break  # Stop if log-likelihood converges

    def _e_step(self, X):
        """
        Perform the expectation step: Calculate the membership probabilities (responsibilities).
        param X: Input data of shape (n_samples, n_features)
        return: Responsibilities matrix of shape (n_samples, n_components)
        """
        n_samples = X.shape[0]
        
        log_prob = np.zeros((n_samples, self.n_components))

        for i in range(self.n_components):
            # Log of weight + log of probability density
            log_prob[:, i] = np.log(self.weights[i]) + self._multivariate_gaussian(X, self.means[i], self.covariances[i])

        # Responsibilities are computed as exp of the normalized log-probabilities
        log_prob -= logsumexp(log_prob,axis=1,keepdims=True)
        
        # Return the responsibilities matrix (used in the M-step)
        return np.exp(log_prob)
    
    def _m_step(self, X, responsibilities):
        """
        Perform the maximization step: Update the model parameters.
        param X: Input data of shape (n_samples, n_features)
        param responsibilities: Responsibilities matrix from the E-step
        """
        n_samples, n_features = X.shape  
    
        # sum of responsibilities for each component (soft counts for each Gaussian component)
        nk = np.sum(responsibilities, axis=0) + 1e-6  # Add small value for numerical stability(regularisation)
        
        # update weights (pie_k): The fraction of points assigned to each Gaussian component
        self.weights = nk / n_samples  
        
        # update means (meu_k): ewighted average of data points for each component
        # np.dot(responsibilities.T, X) computes the weighted sum of data points for each component
        # nk[:, np.newaxis] is used to broadcast nk across each feature dimension for division
        self.means = np.dot(responsibilities.T, X) / nk[:, np.newaxis]
        
        # Initialize covariance matrices for each component
        self.covariances = np.zeros((self.n_components, n_features, n_features))  # Shape: (n_components, n_features, n_features)
        
        # Update covariances (sigma_k) for each component
        for k in range(self.n_components):
            # Calculate the difference between the data points and the mean of component k
            diff = X - self.means[k]
            
            # Compute the weighted covariance matrix for component k
            # responsibilities[:, k][:, np.newaxis] ensures proper broadcasting for element-wise multiplication
            # The covariance is essentially the weighted sum of outer products of (X - mean)
            self.covariances[k] = (responsibilities[:, k][:, np.newaxis] * diff).T @ diff / nk[k]
            
            # Add a small regularization term (1e-6) to the diagonal for numerical stability
            # This prevents the covariance matrix from becoming singular (non-invertible)
            self.covariances[k] += np.eye(n_features) * 1e-6
    def _multivariate_gaussian(self, X, mean, covariance):
        """
        Compute the multivariate Gaussian distribution for the dataset X with given mean and covariance.
        param X: Input data of shape (n_samples, n_features)
        param mean: Mean vector of shape (n_features,)
        param covariance: Covariance matrix of shape (n_features, n_features)
        return: Probability density values for each sample
        """
        n_features = X.shape[1]
        
        # Add a small regularization term to the diagonal of the covariance matrix for numerical stability
        epsilon = 1e-6
        covariance += np.eye(n_features) * epsilon
        
        # compute the difference between each sample and the mean
        diff = X - mean
        
        # compute the exponent part of the multivariate Gaussian PDF formula
        exponent = np.einsum('ij,jk,ik->i', diff, np.linalg.inv(covariance), diff)
        
        # compute the log determinant of the covariance matrix
        log_det_covariance = np.log(np.linalg.det(covariance) + epsilon)
        
        # compute the log of the Gaussian probability density for each sample
        log_prob = -0.5 * (exponent + n_features * np.log(2 * np.pi) + log_det_covariance)
        
        # Return the  log-probabilities
        return multivariate_normal.logpdf(X, mean=mean, cov=covariance)

    def getParams(self):
        """
        Return the current parameters of the GMM (means, covariances, weights).
        """
        return {'means': self.means, 'covariances': self.covariances, 'weights': self.weights}
    
    def getMembership(self, X):
        """
        Return the membership probabilities (responsibilities) for the input data X.
        param X: Input data of shape (n_samples, n_features)
        return: Responsibilities matrix of shape (n_samples, n_components)
        """
        return self._e_step(X)  # Call the E-step function to get the responsibilities
    
    def getLikelihood(self, X):
        """
        Compute and return the overall log-likelihood of the dataset under the current model parameters.
        param X: Input data of shape (n_samples, n_features)
        return: Log-likelihood of the data
        """
        # Initialize the likelihood
        likelihood = np.zeros((X.shape[0], self.n_components))
        
        # For each component, accumulate the weighted likelihood of the data
        for i in range(self.n_components):
            likelihood[:, i] = np.log(self.weights[i]) + self._multivariate_gaussian(X, self.means[i], self.covariances[i])
        
        # Return the sum of log-likelihoods across all samples
        return np.sum(logsumexp(likelihood,axis=1))

    def save_params(self, file_path):
        """
        Save the GMM parameters (means, covariances, weights) to a file.
        param file_path: Path to the file where parameters will be saved
        """
        with open(file_path, 'w') as f:
            f.write("GMM Parameters:\n")
            
            # Save the means
            f.write("Means:\n")
            np.savetxt(f, self.means, fmt='%0.6f')
            
            # Save the covariance matrices
            f.write("\nCovariances:\n")
            for cov in self.covariances:
                np.savetxt(f, cov, fmt='%0.6f')
                f.write("\n")  # Separate each covariance matrix with a newline
            
            # Save the weights
            f.write("Weights:\n")
            np.savetxt(f, self.weights, fmt='%0.6f')
            
    def compute_log_likelihood(self, X):
        """
        Compute and return the log-likelihood of the dataset under the current model parameters.
        param X: Input data of shape (n_samples, n_features)
        return: Log-likelihood of the data
        """
        log_likelihood = np.zeros(X.shape[0])

        for i in range(self.n_components):
            # Add log of weights and log-probabilities
            log_likelihood += np.log(self.weights[i]) + self._multivariate_gaussian(X, self.means[i], self.covariances[i])

        # Sum over all samples
        return np.sum(log_likelihood)
    
    def compute_aic(self, X):
        """Compute Akaike Information Criterion (AIC)."""
        # Compute the log-likelihood
        log_likelihood = self.getLikelihood(X)
        
        n_samples, n_features = X.shape
        
        n_parameters = self.n_components * n_features  # Means
        n_parameters += self.n_components * n_features * (n_features + 1) / 2  # Covariances
        n_parameters += self.n_components  # Weights
        
        # AIC = 2 * number of parameters - 2 * log-likelihood
        aic = 2 * n_parameters - 2 * log_likelihood
        return aic
    
    def compute_bic(self, X):
        """Compute Bayesian Information Criterion (BIC)."""
        # Compute the log-likelihood
        log_likelihood = self.getLikelihood(X)
        
        n_samples, n_features = X.shape
        
        n_parameters = self.n_components * n_features  # Means
        n_parameters += self.n_components * n_features * (n_features + 1) / 2  # Covariances
        n_parameters += self.n_components # Weights
        
        # BIC = log(n_samples) * number of parameters - 2 * log-likelihood
        bic = np.log(n_samples) * n_parameters - 2 * log_likelihood
        return bic
    
    def ClusterAssignments(self, X):
        responsibilities = self.getMembership(X)
        return np.argmax(responsibilities, axis=1)
