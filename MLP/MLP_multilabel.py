import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss


# Activation functions
def sigmoid(x):
    x = np.clip(x,-500,500)
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def linear(x):
    return x

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def binary_cross_entropy(y_true, y_pred):
    # To avoid log(0) which is undefined, clip the predicted values between a small epsilon value
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid log(0)
    
    # Binary Cross-Entropy
    bce_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    return bce_loss

# Derivatives of activation functions
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def linear_derivative(x):
    return 1

def softmax_derivative(output):
    # Softmax derivative is more complex and is usually handled differently in practice.
    return output * (1 - output)

def binary_cross_entropy_derivative(y_true, y_pred):
    # To avoid division by zero, clip the predicted values between a small epsilon value
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid division by 0
    
    bce_derivative = (y_pred - y_true) / (y_pred * (1 - y_pred))
    
    return bce_derivative

# Loss function (Categorical Cross-Entropy)
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    return 2*(y_pred - y_true)

def rmse_loss(y_true, y_pred):
    return np.sqrt(mse_loss(y_true, y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def mean_absolute_error(y_true,y_pred):
    return np.mean(np.abs(y_true-y_pred))

class Optimizer:
    def update(self, weights, gradients, learning_rate):
        pass

class SGD(Optimizer):
    def update(self, weights, gradients, learning_rate):
        #print(weights - learning_rate * gradients)
        #print(gradients)
        return weights - learning_rate * gradients
    
    def get_batch_size(self, n_samples):
        return 1

class BatchGD(Optimizer):
    def update(self, weights, gradients, learning_rate):
        return weights - learning_rate * gradients

    def get_batch_size(self, n_samples):
        return n_samples

class MiniBatchGD(Optimizer):
    def __init__(self, batch_size=32):
        self.batch_size = batch_size

    def update(self, weights, gradients, learning_rate):
        return weights - learning_rate * gradients

    def get_batch_size(self, n_samples):
        return min(self.batch_size, n_samples)
    
class MLP_multilabel:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01,activation='relu', optimizer='sgd', batch_size=32, epochs=100):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Activation functions and their derivatives
        self.activations = {
            'sigmoid': (sigmoid, sigmoid_derivative),
            'tanh': (tanh, tanh_derivative),
            'relu': (relu, relu_derivative),
            'linear': (linear, linear_derivative),
            'binary_cross_entropy':(binary_cross_entropy,binary_cross_entropy_derivative)
        }
        self.activation, self.activation_derivative = self.activations[activation]
        
        # Optimizers
        self.optimizers = {
            'sgd': SGD(),
            'batchgd': BatchGD(),
            'minibatchgd': MiniBatchGD()
        }
        self.optimizer = self.optimizers[optimizer]
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self._initialize_weights()
        
        
    def _initialize_weights(self):
        np.random.seed(42)  # for reproducibility
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01)
            self.biases.append(np.zeros((1, layer_sizes[i+1])))

    def _forward(self, X):
        activations = [X]
        pre_activations = []
        # print(activations[-1])
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(activations[-1], w) + b
            pre_activations.append(z)
            # print(activations[-1])
            if i == len(self.weights) - 1:
                # For regression tasks, use linear activation in the output layer
                activations.append(sigmoid(z))
            else:
                activations.append(self.activation(z))
        # print(activation)
        return activations, pre_activations

    def _backward(self, X, y, activations, pre_activations):
        gradients_w = []
        gradients_b = []
        #print((y,activations[-1]))
        delta = binary_cross_entropy_derivative(y, activations[-1])*sigmoid_derivative(pre_activations[-1]) #last layer is for prediction of classes
        #print(delta)
        # print("Initial delta (output layer):", delta)
        for i in reversed(range(len(self.weights))):
            gradients_w.append(np.dot(activations[i].T, delta))
            gradients_b.append(np.sum(delta, axis=0, keepdims=True))
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(pre_activations[i-1])
            # print("Layer:", i, "Gradient W:", gradients_w[-1], "Gradient B:", gradients_b[-1])

        gradients_w.reverse()
        gradients_b.reverse()
        return gradients_w, gradients_b

    def fit(self, X, y, X_val=None, y_val=None):
        batch_size_1 = self.optimizer.get_batch_size(X.shape[0])
        for epoch in range(self.epochs):
            epoch_loss = 0
            for i in range(0, X.shape[0], batch_size_1):
                X_batch = X[i:i+self.batch_size]
                y_batch = y[i:i+self.batch_size]
                activations, pre_activations = self._forward(X_batch)
                # print(activations[-1],pre_activations[-1])
                gradients_w, gradients_b = self._backward(X_batch, y_batch, activations, pre_activations)
                for j in range(len(self.weights)):
                    self.weights[j] = self.optimizer.update(self.weights[j], gradients_w[j], self.learning_rate)
                    self.biases[j] = self.optimizer.update(self.biases[j], gradients_b[j], self.learning_rate)
                epoch_loss += binary_cross_entropy(y_batch, activations[-1])
                
            train_predict = self.predict_probability(X)
            train_loss = binary_cross_entropy(y,train_predict)
            #print(train_predict,X)
            # Early stopping check (optional based on validation set)
            if X_val is not None and y_val is not None:
                val_predictions = self.predict(X_val)
                val_loss = binary_cross_entropy(y_val, self.predict_probability(X_val))
                accuracy,precision,recall,f1,hamming = self.evaluate_model(y_val, val_predictions)
                # Log validation metrics to W&B
                # wandb.log({
                #     "epoch": epoch + 1,
                #     "val_loss": val_loss,
                #     "accuracy": accuracy,
                #     "precision": precision,
                #     "recall" : recall,
                #     "f1":f1,
                #     "hamming loss":hamming
                # })

            #print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')

    def predict_probability(self, X):
        activations, _ = self._forward(X) 
        # print(activations[-1])          
        return activations[-1]

    def predict(self, X):
        probability = self.predict_probability(X)
        #print(probability)
        return (probability > 0.5).astype(int)
        #return probability
    
    def gradient_check(self, X, y, epsilon=1e-4):
        """
        Performs numerical gradient checking to compare backpropagation gradients with numerical gradients.
        Args:
            X: Input data
            y: Target labels
            epsilon: Small value for computing numerical gradients (default: 1e-7)
        
        Returns:
            numerical_grads_w: Numerical gradients for weights
            numerical_grads_b: Numerical gradients for biases
        """
        numerical_grads_w = []
        numerical_grads_b = []
        
        # Save the original parameters (weights and biases)
        original_weights = [w.copy() for w in self.weights]
        original_biases = [b.copy() for b in self.biases]
        
        # Check weights
        for l in range(len(self.weights)):
            grad_w = np.zeros_like(self.weights[l])
            for i in range(self.weights[l].shape[0]):
                for j in range(self.weights[l].shape[1]):
                    original_value = self.weights[l][i, j]
                    
                    # f(x + epsilon)
                    self.weights[l][i, j] = original_value + epsilon
                    loss_plus_epsilon = binary_cross_entropy(y,self.predict_probability(X))  # Compute loss with perturbed weight mse
                    
                    # f(x - epsilon)
                    self.weights[l][i, j] = original_value - epsilon
                    loss_minus_epsilon = binary_cross_entropy(y,self.predict_probability(X))  # Compute loss with perturbed weight
                    
                    # gradient approximation
                    grad_w[i, j] = (loss_plus_epsilon - loss_minus_epsilon) / (2 * epsilon)
                    # print(grad_w[i, j] , original_value)
                    # Reset to original value
                    self.weights[l][i, j] = original_value  

            numerical_grads_w.append(grad_w)
        
        # Check biases
        for l in range(len(self.biases)):
            grad_b = np.zeros_like(self.biases[l])
            for i in range(self.biases[l].shape[1]):
                original_value = self.biases[l][0, i]
                
                # f(x + epsilon)
                self.biases[l][0, i] = original_value + epsilon
                loss_plus_epsilon = binary_cross_entropy(y,self.predict_probability(X))
                
                # f(x - epsilon)
                self.biases[l][0, i] = original_value - epsilon
                loss_minus_epsilon = binary_cross_entropy(y,self.predict_probability(X))
                
                # Gradient approximation
                grad_b[0, i] = (loss_plus_epsilon - loss_minus_epsilon) / (2 * epsilon)
                
                # Reset to original value
                self.biases[l][0, i] = original_value  

            numerical_grads_b.append(grad_b)
        
        return numerical_grads_w, numerical_grads_b
    
    def compare_gradients(self, X, y):
        """
        Compares the gradients computed via backpropagation with the numerical gradients.
        """
        # forward and backward propagation to get gradients from backpropagation
        activations, pre_activations = self._forward(X)
        backprop_grads_w, backprop_grads_b = self._backward(X, y, activations, pre_activations)
        numerical_grads_w, numerical_grads_b = self.gradient_check(X, y)
        # compare gradients
        for l in range(len(self.weights)):
            print(f"Layer {l+1} weights gradient difference:")
            # print(backprop_grads_w[l][0],numerical_grads_w[l][0])
            diff_w = np.linalg.norm(backprop_grads_w[l] - numerical_grads_w[l]) / np.linalg.norm(backprop_grads_w[l] + numerical_grads_w[l])
            print(f"Relative Difference (weights): {diff_w}")
        
        for l in range(len(self.biases)):
            print(f"Layer {l+1} biases gradient difference:")
            diff_b = np.linalg.norm(backprop_grads_b[l] - numerical_grads_b[l]) / np.linalg.norm(backprop_grads_b[l] + numerical_grads_b[l])
            print(f"Relative Difference (biases): {diff_b}")
            
    def evaluate_model(self,y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='micro')
        recall = recall_score(y_true, y_pred, average='micro')
        f1 = f1_score(y_true, y_pred, average='micro')
        hamming = hamming_loss(y_true, y_pred)

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"Hamming Loss: {hamming}")
        
        return accuracy,precision,recall,f1,hamming
