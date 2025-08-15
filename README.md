# Machine Learning Models Implemented from Scratch

This repository contains various machine learning models, each implemented from scratch using Python and NumPy. Every model is organized in its own folder, and each implementation follows a class-based structure. Training, validation, and testing routines are clearly separated within each class.

## Table of Contents
- [Linear Regression](#linear-regression)
- [K-Means Clustering](#k-means-clustering)
- [Gaussian Mixture Model (GMM)](#gaussian-mixture-model-gmm)
- [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
- [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
- [Multi-Layer Perceptron (MLP)](#multi-layer-perceptron-mlp)
- [Autoencoders](#autoencoders)
- [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)

---

## Linear Regression

**Location:** `linear-regression/linear_regression.py`

A simple linear regression model using gradient descent for parameter optimization.

- **Features:**
  - Polynomial feature expansion
  - Gradient descent for weights and bias
  - Mean Squared Error cost function

**Usage Example:**
```python
from linear_regression import Linear_Regression
model = Linear_Regression(learning_rate=0.01, no_of_iterations=1000)
model.fit(X, Y)
predictions = model.predict(X)
```

---

## K-Means Clustering

**Location:** `k_mean_clustering/k_mean_clustering.py`

Implements the K-Means clustering algorithm for unsupervised learning.

- **Features:**
  - Random centroid initialization
  - Iterative assignment and update steps
  - Supports custom number of clusters

**Usage Example:**
```python
from k_mean_clustering import KMeans
model = KMeans(n_clusters=3)
model.fit(X)
labels = model.predict(X)
```

---

## Gaussian Mixture Model (GMM)

**Location:** `GMM/GMM.py`

A probabilistic model for representing normally distributed subpopulations within an overall population.

- **Features:**
  - Expectation-Maximization (EM) algorithm
  - Soft clustering
  - Supports multiple Gaussian components

**Usage Example:**
```python
from GMM import GMM
model = GMM(n_components=2)
model.fit(X)
labels = model.predict(X)
```

---

## Principal Component Analysis (PCA)

**Location:** `pca/pca.py`

Dimensionality reduction technique that projects data onto principal components.

- **Features:**
  - Eigen decomposition of covariance matrix
  - Explained variance calculation
  - Data transformation to lower dimensions

**Usage Example:**
```python
from pca import PCA
model = PCA(n_components=2)
model.fit(X)
X_reduced = model.transform(X)
```

---

## K-Nearest Neighbors (KNN)

**Location:** `knn/knn.py`

Implements both basic and optimized KNN classifiers.

- **Features:**
  - Choice of distance metric: Euclidean or Manhattan
  - Vectorized distance computation (OptimizedKNN)
  - Metrics for accuracy, precision, recall, F1-score

**Usage Example:**
```python
from knn import KNN, OptimizedKNN
model = KNN(k=3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

## Multi-Layer Perceptron (MLP)

**Location:** `MLP/`

Includes several variants:
- `MLP_BCE.py`: Binary cross-entropy loss
- `MLP_classifier_regressor.py`: Classification and regression
- `MLP_multilabel.py`: Multi-label classification
- `MLP_regression.py`: Regression
- `MLP_soft_max.py`: Softmax output

- **Features:**
  - Fully connected layers
  - Custom activation functions
  - Support for multiple loss functions

**Usage Example:**
```python
from MLP_classifier_regressor import MLP
model = MLP(hidden_layers=[64, 32])
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

## Autoencoders

**Location:** `Auto_Encoders/`

Includes:
- `AutoEncoders.py`: Basic autoencoder
- `cnn-autoencoder.py`: Convolutional autoencoder
- `pca-autoencoder.py`: PCA-based autoencoder

- **Features:**
  - Dimensionality reduction
  - Reconstruction loss
  - Support for both fully connected and convolutional architectures

**Usage Example:**
```python
from AutoEncoders import AutoEncoder
model = AutoEncoder(input_dim=784, hidden_dim=64)
model.fit(X_train)
reconstructed = model.predict(X_test)
```

---

## Convolutional Neural Network (CNN)

**Location:** `cnn/cnn.py`, `cnn/multilabel-cnn.py`

Implements basic and multi-label CNNs for image classification.

- **Features:**
  - Convolutional and pooling layers
  - Multi-label support
  - Custom training and evaluation routines

**Usage Example:**
```python
from cnn import CNN
model = CNN()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

## Notes

- All models are implemented from scratch using only NumPy and standard Python libraries.
- Each model is self-contained and can be used independently.
- Training, validation, and testing routines are clearly separated for educational clarity.

---

For more details, refer to the individual Python files in each folder. Each file contains class definitions, method documentation, and example usage.
# Make predictions
```
predictions = model.predict(X)
print(predictions)
```

## Mathematical Formula

The linear regression model aims to minimize the cost function, which measures the error between the predicted values and the actual values. The model parameters (weights \( w \) and bias \( b \)) are updated using the following gradient descent equations:

### Hypothesis Function

\[$ \hat{Y} = Xw + b $]

### Cost Function (Mean Squared Error)

\[$ J(w, b) = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{Y}_i - Y_i \right)^2 \ $]

### Gradient Descent Update Rules

- **Weight Update**:
  
  \[$ w := w - \alpha \cdot \frac{2}{m} \cdot X^T \cdot (\hat{Y} - Y) \ $]

- **Bias Update**:
  
  \[$ b := b - \alpha \cdot \frac{2}{m} \cdot \sum_{i=1}^{m} (\hat{Y}_i - Y_i) \ $]

### Where:

- \($ \alpha \ $) is the learning rate.
- \( m \) is the number of training examples.
- \( X \) is the input feature matrix.
- \( Y \) is the actual output vector.
- \($ \hat{Y} \ $) is the predicted output vector.


Code Structure
Here’s how the code handles these:

Feature Expansion: The _expand_features method generates polynomial features from the input data. It raises each element of the input to different powers based on the specified degree and creates a new array of these polynomial features.

Weight Updates: The update_weights method updates each weight in w based on the gradient of the loss function. The update rules apply gradient descent to adjust each coefficient in the weight vector w to minimize the mean squared error.

Prediction: The predict method uses the expanded features and the current weights (including bias) to make predictions. This involves calculating the dot product of the polynomial features and the weights and adding the bias.



# KNN Classifier

This section contains a basic implementation of the k-nearest neighbors (KNN) algorithm, as well as an optimized version for efficiency. Additionally, it includes a set of metrics to evaluate the performance of the classifiers.


## Table of Contents
- [KNN](#knn)
- [OptimizedKNN](#optimizedknn)
- [Metrics](#metrics)
- [Usage](#usage_knn)
## KNN

The `KNN` class implements a basic k-nearest neighbors classifier. It can take different values of `k` and the distance metric as arguments.

### Methods

- **`__init__(self, k, distance_metric='euclidean')`**

  Initializes the KNN classifier with:
  - `k`: Number of nearest neighbors to consider.
  - `distance_metric`: Metric used to calculate the distance ('euclidean' or 'manhattan').

- **`fit(self, X_train, y_train)`**

  Stores the training data:
  - `X_train`: Feature matrix for training data.
  - `y_train`: Labels for training data.

- **`_compute_distances(self, x)`**

  Computes the distances between a test instance `x` and all training instances using the selected distance metric by calculating the distance of point `x` from all points in the training dataset sequentially. This means calculating the distance of each point in the test dataset with every row of the train set and finding the nearest `k` distances for each test point.

- **`predict(self, X_test)`**

  Predicts labels for the test instances in `X_test`:
  - Computes distances between each test instance and all training instances.
  - Identifies the `k` nearest neighbors/labels by sorting according to the distance and returning the index of `k` nearest neighbors.
  - Determines the most common label among these neighbors and returns it.

## OptimizedKNN

The `OptimizedKNN` class is an optimized version of the KNN classifier that uses vectorized operations for distance computation, which can compute the distances in parallel instead of sequentially as in the basic KNN.

### Methods

- **`__init__(self, k, distance_metric='euclidean')`**

  Initializes the optimized KNN classifier with:
  - `k`: Number of nearest neighbors to consider.
  - `distance_metric`: Metric used to calculate the distance ('euclidean' or 'manhattan').

- **`fit(self, X_train, y_train)`**

  Stores the training data:
  - `X_train`: Feature matrix for training data.
  - `y_train`: Labels for training data.

- **`_compute_distances(self, x)`**

  Computes distances between a test instance `x` and all training instances using vectorized operations:
  - `x`: A single data point from the test set.
  - Returns a numpy array of distances.
  - For Euclidean distance calculation:
    - `np.linalg.norm(..., axis=1)` calculates the Euclidean norm (or L2 norm) along the specified axis.
    - Here, `axis=1` indicates that the norm should be computed across columns for each row.
    - This results in a 1D array of distances where each entry represents the distance between `x` and a training instance.This is much better than the original KNN since in there for each point in x_test  distance was calculated with x_train.

- **`predict(self, X_test)`**

  Predicts labels for the test instances in `X_test`:
  - Computes distances between each test instance and all training instances.
  - Identifies the `k` nearest neighbors.
  - Determines the most common label among these neighbors and returns it.

## Metrics

The `Metrics` class provides static methods to calculate various performance metrics.

### Methods

- **`accuracy(y_true, y_pred)`**

  Calculates the accuracy of the predictions:
  - `y_true`: True labels.
  - `y_pred`: Predicted labels.
  - Returns the accuracy score.

- **`precision(y_true, y_pred)`**

  Calculates the precision for each class and averages them:
  - `y_true`: True labels.
  - `y_pred`: Predicted labels.
  - Returns the average precision score.

- **`recall(y_true, y_pred)`**

  Calculates the recall for each class and averages them:
  - `y_true`: True labels.
  - `y_pred`: Predicted labels.
  - Returns the average recall score.

- **`f1_score(y_true, y_pred)`**

  Calculates the F1 score for each class and averages them:
  - `y_true`: True labels.
  - `y_pred`: Predicted labels.
  - Returns the average F1 score.

## Usage_KNN

### Basic KNN Class

```python
import numpy as np
from knn import KNN  # Import the KNN class from the knn module

# Sample dataset
X_train = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [7, 8]])
Y_train = np.array(['A', 'A', 'B', 'B', 'A'])
X_test = np.array([[2, 2], [6, 5]])

# Initialize and train the model
model = KNN(k=3, distance_metric='euclidean')
model.fit(X_train, Y_train)

# Make predictions
predictions = model.predict(X_test)
print("Predictions:", predictions)
```
### OptimizedKNN Class
```python
import numpy as np
from knn import OptimizedKNN  # Import the OptimizedKNN class from the knn module

# Sample dataset
X_train = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [7, 8]])
Y_train = np.array(['A', 'A', 'B', 'B', 'A'])
X_test = np.array([[2, 2], [6, 5]])

# Initialize and train the model
model = OptimizedKNN(k=3, distance_metric='manhattan')
model.fit(X_train, Y_train)

# Make predictions
predictions = model.predict(X_test)
print("Predictions:", predictions)
```
### CHAT-GPT Prompts:
1) Instead of finding distance of each point in x_test with each point in x_train how can we use vectorization to reduce the time complexity
2) How to create a GIF from multiple plots generated during iterations of some predictive process