import numpy as np
from numpy.random import uniform

class Kmeans:
    def __init__(self, n_clusters, iterations, tol=1e-4):
        """
        Initialize the KMeans class with the number of clusters (k), maximum iterations, tolerance.

        param n_clusters: Number of clusters
        param iterations: Maximum number of iterations to run the algorithm
        param tol: Tolerance for convergence. If the centroids do not change more than this, stop the algorithm.
        """
        self.n_clusters = n_clusters
        self.iterations = iterations
        self.centroids = None
        self.tol = tol
        
    def fit(self, X_train):
        X_train = np.array(X_train)
        
        #self.centroids = X_train[np.random.choice(X_train.shape[0], self.n_clusters, replace=False)]
        self.centroids = [X_train[i] for i in range(self.n_clusters)]
                
        # Loop through the maximum number of iterations
        for _ in range(self.iterations):
            #Assign points to the nearest centroid
            labels = []
            for point in X_train:
                distances = []
                for centroid in self.centroids:
                    # Calculate Euclidean distance between the point and the centroid
                    distance = np.sqrt(np.sum((point - centroid) ** 2))
                    distances.append(distance)
                # Assign the point to the nearest centroid
                labels.append(np.argmin(distances))
            
            # Step 2: Recompute centroids by averaging points in each cluster
            new_centroids = []
            for i in range(self.n_clusters):
                # Get all points assigned to the cluster `i`
                points_in_cluster = [X_train[j] for j in range(len(X_train)) if labels[j] == i]
                
                if len(points_in_cluster) > 0:
                    # Calculate the new centroid as the mean of points in the cluster
                    new_centroid = np.mean(points_in_cluster, axis=0)
                else:
                    # If no points were assigned to the cluster, keep the old centroid
                    new_centroid = self.centroids[i]
                
                new_centroids.append(new_centroid)
            
            # Check for convergence
            max_shift = max(np.sqrt(np.sum((np.array(new_centroids) - np.array(self.centroids)) ** 2, axis=1)))
            self.centroids = new_centroids
            if max_shift <= self.tol:
                break
    
    def get_cost(self, X_train):
        """
        Compute the Within-Cluster Sum of Squares (WCSS), which represents the total cost.

        param X_train: Input data (512-dimensional embeddings)
        return: The total WCSS (cost)
        """
        X_train = np.array(X_train)
        labels = []
        wcss = 0

        # Assign each point to the nearest centroid
        for point in X_train:
            distances = []
            for centroid in self.centroids:
                # Calculate Euclidean distance between the point and the centroid
                distance = np.sqrt(np.sum((point - centroid) ** 2))
                distances.append(distance)
            # Assign the point to the nearest centroid
            nearest_centroid = np.argmin(distances)
            labels.append(nearest_centroid) 
        
        # Calculate the WCSS (Within-Cluster Sum of Squares)
        for i in range(self.n_clusters):
            points_in_cluster = np.array([X_train[j] for j in range(len(X_train)) if labels[j] == i])
            for point in points_in_cluster:
                wcss += np.sum((point - self.centroids[i]) ** 2)        
        return wcss
    
    def predict(self, X_test):
        """
        Predict the cluster label for each point in X_test based on the centroids determined by the fit() function.

        param X_test: Input data to assign to clusters
        return: List of cluster labels for each data point in X_test
        """
        X_test = np.array(X_test)
        labels = []
        
        for point in X_test:
            distances = []
            for centroid in self.centroids:
                # Calculate Euclidean distance between the point and the centroid
                distance = np.sqrt(np.sum((point - centroid) ** 2))
                distances.append(distance)
            # Assign the point to the nearest centroid
            nearest_centroid = np.argmin(distances)
            labels.append(nearest_centroid)
        
        return labels
