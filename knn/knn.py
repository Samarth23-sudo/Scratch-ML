import numpy as np
import pandas as pd
from collections import Counter

class KNN:
    def __init__(self, k, distance_metric='euclidean'):
        """
        Initialize KNN with the given k and distance metric.
        """
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X_train, y_train):
        """
        Store the training data.
        """
        self.X_train = X_train
        self.y_train = y_train

    def _compute_distances(self, x):
        """
        Compute distances between the input and all training points.
        """
        distances=[]
        for x_train in self.X_train:
            if self.distance_metric == 'euclidean':
                dist = np.sqrt(np.sum((x_train - x) ** 2))
            elif self.distance_metric == 'manhattan':
                dist = np.sum(np.abs(x_train - x))
            else:
                raise ValueError(f"Unknown distance metric: {self.distance_metric}")
            distances.append(dist)
        return distances

    def predict(self, X_test):
        """
        Predict the track_genre for each instance in X_test.
        """
        predictions = []
        i=0
        
        for x in X_test:
            distances = self._compute_distances(x)
            # print(predicted_distances.shape)
            # Get the indices of the k-nearest neighbors
            neighbor_indices = np.argsort(distances)[:self.k]
            # Get the most common class among these neighbors
            neighbor_genres = [self.y_train[i] for i in neighbor_indices]  
            print(neighbor_genres)
            most_common_genre = Counter(neighbor_genres).most_common(1)[0][0]         
            predictions.append(most_common_genre)
            # print(y_test[i],most_common_genre)
            # i+=1
        return predictions


class OptimizedKNN:
    def __init__(self, k, distance_metric='euclidean'):
        """
        Initialize KNN with the given k and distance metric.
        k: Number of nearest neighbors to consider for classification.
        distance_metric: Distance metric to use ('euclidean' or 'manhattan').
        """
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X_train, y_train):
        """
        Store the training data for later use in prediction.
        X_train: Training feature set.
        y_train: Training labels.
        """
        self.X_train = X_train
        self.y_train = y_train

    def _compute_distances(self, x):
        """
        Compute distances between the input and all training points using vectorized operations.
        x: A single data point from the test set.
        Returns: A numpy array of distances.
        """
        # Reshape x to ensure it's a 2D array for parallel computation with X_train dataset 
        x = list(x)
        if self.distance_metric == 'euclidean':
            # Vectorized Euclidean distance computation
            distances = np.linalg.norm(self.X_train - x, axis=1)
        elif self.distance_metric == 'manhattan':
            # Vectorized Manhattan distance computation
            distances = np.sum(np.abs(self.X_train - x), axis=1)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        return distances

    def predict(self, X_test):
        """
        Predict the track_genre for each instance in X_test using the k-nearest neighbors.
        X_test: Test feature set.
        Returns: A list of predicted genres.
        """
        predictions = []

        for x in X_test:
            # Compute the distance from the test point to all training points
            distances = self._compute_distances(x)
            
            # Get the indices of the k-nearest neighbors (smallest distances)
            neighbor_indices = np.argsort(distances)[:self.k]
            
            # Retrieve the labels of the k-nearest neighbors
            neighbor_genres = [self.y_train[i] for i in neighbor_indices]
            # Determine the most common genre among the neighbors
            most_common_genre = Counter(neighbor_genres).most_common(1)[0][0]
            
            # Append the predicted genre to the list of predictions
            predictions.append(most_common_genre)
        
        return predictions


# class Metrics:
#     @staticmethod
#     def accuracy(y_true, y_pred):
#         """
#         Calculate the accuracy.
#         """
#         return np.sum(y_true == y_pred) / len(y_true)

#     @staticmethod
#     def precision(y_true, y_pred):  
#         """
#         Calculate the precision for each class and average them.
#         """
#         classes = np.unique(y_true)
#         precisions = []
#         for cls in classes:
#             true_positives=0
#             predicted_positives=0
#             for i in range(len(y_pred)):
#                 if (y_true[i] == cls) & (y_pred[i] == cls):
#                     true_positives+=1
#                 if(y_pred[i] == cls):
#                     predicted_positives+=1
#             precisions.append(true_positives / predicted_positives if predicted_positives != 0 else 0)
#         return np.mean(precisions)

#     @staticmethod
#     def recall(y_true, y_pred):
#         """
#         Calculate the recall for each class and average them.
#         """
#         classes = np.unique(y_true)
#         recalls = []
#         for cls in classes:
#             true_positives=0
#             actual_positives=0
#             for i in range(len(y_pred)):
#                 if (y_true[i] == cls) & (y_pred[i] == cls):
#                     true_positives+=1
#                 if(y_true[i]== cls):
#                     actual_positives+=1
#             recalls.append(true_positives / actual_positives if actual_positives != 0 else 0)
#         return np.mean(recalls)

#     @staticmethod
#     def f1_score(y_true, y_pred):
#         """
#         Calculate the F1 score for each class and average them.
#         """
#         precision = Metrics.precision(y_true, y_pred)
#         recall = Metrics.recall(y_true, y_pred)
#         return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0