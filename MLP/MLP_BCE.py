import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt


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
    #print(y_true,y_pred)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid log(0)
    
    # Binary Cross-Entropy
    bce_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce_loss

def binary_cross_entropy_derivative(y_true, y_pred):
    # To avoid division by zero, clip the predicted values between a small epsilon value
    #print(y_true,y_pred)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid division by 0
    
    bce_derivative = (y_pred - y_true) / (y_pred * (1 - y_pred))
    #print(bce_derivative)
    return bce_derivative

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

# Loss function (Categorical Cross-Entropy)
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    return 2*(y_pred - y_true)

class Optimizer:
    def update(self, weights, gradients, learning_rate):
        pass
# Optimizers
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

class MLPClassifier_BCE:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01,
                 activation='relu', optimizer='sgd', batch_size=32, epochs=100):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_history = []  

        # Activation functions and their derivatives
        self.activations = {
            'sigmoid': (sigmoid, sigmoid_derivative),
            'tanh': (tanh, tanh_derivative),
            'relu': (relu, relu_derivative),
            'softmax': (softmax, softmax_derivative),
            'linear': (linear,linear_derivative),
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
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01)
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

    def _forward(self, X):
        activations = [X]
        pre_activations = []

        for w, b in zip(self.weights, self.biases):
            z = np.dot(activations[-1], w) + b
            pre_activations.append(z)
            activations.append(self.activation(z))

        return activations, pre_activations

    def _backward(self, X, y, activations, pre_activations):
        gradients_w = []
        gradients_b = []

        # Calculate the loss derivative for  output layer
        delta = binary_cross_entropy_derivative(y, activations[-1]) * self.activation_derivative(pre_activations[-1])

        for i in reversed(range(len(self.weights))):
            gradients_w.append(np.dot(activations[i].T, delta)/y.shape[0])
            gradients_b.append(np.sum(delta, axis=0, keepdims=True)/y.shape[0])

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(pre_activations[i - 1])

        gradients_w.reverse()
        gradients_b.reverse()

        return gradients_w, gradients_b

    def fit(self, X, y_one_hot,X_val=None, y_val=None):
        batch_size_1 = self.optimizer.get_batch_size(X.shape[0])
        epoch_loss_list = []
        for epoch in range(self.epochs):
            epoch_loss = 0

            for i in range(0, X.shape[0], batch_size_1):
                X_batch = X[i:i + self.batch_size]
                y_batch = y_one_hot[i:i + self.batch_size]

                activations, pre_activations = self._forward(X_batch)
                gradients_w, gradients_b = self._backward(X_batch, y_batch, activations, pre_activations)

                for j in range(len(self.weights)):
                    self.weights[j] = self.optimizer.update(self.weights[j], gradients_w[j], self.learning_rate)
                    self.biases[j] = self.optimizer.update(self.biases[j], gradients_b[j], self.learning_rate)

                epoch_loss += binary_cross_entropy(y_batch, activations[-1])
                
            
            train_predict = self.predict(X)
            train_acc = self.accuracy(X,y_one_hot)
            train_loss = binary_cross_entropy(y_one_hot,train_predict)
            epoch_loss_list.append(train_loss)
            # Log metrics
            # wandb.log({
            #     "epoch": epoch + 1,
            #     "train_loss": train_loss,
            #     # "train_accuracy": train_acc,
            #     # 'val_f1_score': val_f1,
            #     # 'val_precision': val_precision,
            #     # 'val_recall': val_recall
            # })
            if X_val is not None and y_val is not None:
                val_predictions = self.predict(X_val)
                val_loss = binary_cross_entropy(y_val, val_predictions)
                val_acc = self.accuracy(X_val, y_val)
                val_f1 = self.f1_score(X_val, y_val)
                val_precision = self.precision(X_val,y_val)
                val_recall = self.recall(X_val,y_val)

                # Log validation metrics to W&B
                # wandb.log({
                #     "epoch": epoch + 1,
                #     "val_loss": val_loss,
                #     "val_accuracy": val_acc,
                #     "val_f1_score": val_f1,
                #     "val_precision": val_precision,
                #     "val_recall": val_recall
                # })
                #print(f'Epoch {epoch + 1}, Validation Loss: {val_loss}, Accuracy: {val_acc}')
                
            print(f'Epoch {epoch + 1}, BCE_Loss: {epoch_loss/self.batch_size}')
        self.loss_history.append(epoch_loss_list)  # Store loss history for each run    
        
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
                    loss_plus_epsilon = binary_cross_entropy(y,self.predict(X))  # Compute loss with perturbed weight bce
                    
                    
                    # f(x - epsilon)
                    self.weights[l][i, j] = original_value - epsilon
                    loss_minus_epsilon = binary_cross_entropy(y,self.predict(X))  # Compute loss with perturbed weight
                    
                    # gradient approximation
                    grad_w[i, j] = (loss_plus_epsilon - loss_minus_epsilon) / (2 * epsilon)
                    # print(grad_w[i, j] , original_value)
                    # Reset to original value
                    #print(grad_w[i, j],original_value)
                    self.weights[l][i, j] = original_value  

            numerical_grads_w.append(grad_w)
        
        # Check biases
        for l in range(len(self.biases)):
            grad_b = np.zeros_like(self.biases[l])
            for i in range(self.biases[l].shape[1]):
                original_value = self.biases[l][0, i]
                
                # f(x + epsilon)
                self.biases[l][0, i] = original_value + epsilon
                loss_plus_epsilon = binary_cross_entropy(y,self.predict(X))
                
                # f(x - epsilon)
                self.biases[l][0, i] = original_value - epsilon
                loss_minus_epsilon = binary_cross_entropy(y,self.predict(X))
                
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


    def predict(self, X):
        activations, _ = self._forward(X)
        #print(activations[-1])
        return activations[-1]  # Return the softmax probabilities

    def accuracy(self, X, y_true):
        y_pred_probs = self.predict(X)
        if y_true.shape[1] == 1:
            y_pred_classes = np.round(y_pred_probs).astype(int)
            y_true_classes = y_true  # Assuming y_true is not one-hot encoded for binary classification
        else:
            y_pred_classes = np.argmax(y_pred_probs, axis=1)
            y_true_classes = np.argmax(y_true, axis=1)  # Convert one-hot encoded true labels back to class indices
        accuracy = np.mean(y_pred_classes == y_true_classes)
        return accuracy
    
    def precision(self,X,y_true):
        y_pred_probs = self.predict(X)
        
        if y_pred_probs.shape[1] == 1:
            y_pred_classes = np.round(y_pred_probs).astype(int)
            y_true_classes = y_true
        else:
            y_pred_classes = np.argmax(y_pred_probs, axis=1)
            y_true_classes = np.argmax(y_true, axis=1)
        
        classes = np.unique(y_true_classes)
        # print(y_pred_classes,y_true_classes,classes)
        precisions = []
        for cls in classes:
            true_positives = 0
            predicted_positives = 0
            for i in range(len(y_pred_classes)):
                if (y_true_classes[i] == cls) & (y_pred_classes[i] == cls):
                    true_positives += 1
                if y_pred_classes[i] == cls:
                    predicted_positives += 1
            precisions.append(true_positives / predicted_positives if predicted_positives != 0 else 0)
        return np.mean(precisions)
    
    def recall(self,X,y_true):
        y_pred_probs = self.predict(X)
        
        if y_pred_probs.shape[1] == 1:
            y_pred_classes = np.round(y_pred_probs).astype(int)
            y_true_classes = y_true
        else:
            y_pred_classes = np.argmax(y_pred_probs, axis=1)
            y_true_classes = np.argmax(y_true, axis=1)
        
        classes = np.unique(y_true_classes)
        # print(y_pred_classes,y_true_classes,classes)
        recalls = []
        for cls in classes:
            true_positives = 0
            actual_positives = 0
            for i in range(len(y_pred_classes)):
                if (y_true_classes[i] == cls) & (y_pred_classes[i] == cls):
                    true_positives += 1
                if y_true_classes[i] == cls:
                    actual_positives += 1
            recalls.append(true_positives / actual_positives if actual_positives != 0 else 0)
        return np.mean(recalls)
    
    def f1_score(self,X,y_true, average='macro'):
        y_pred_probs = self.predict(X)
        
        if y_pred_probs.shape[1] == 1:
            y_pred_classes = np.round(y_pred_probs).astype(int)
            y_true_classes = y_true
        else:
            y_pred_classes = np.argmax(y_pred_probs, axis=1)
            y_true_classes = np.argmax(y_true, axis=1)
        
        precision = self.precision(X,y_true)
        recall = self.recall(X,y_true)
        
        if average == 'macro':
            macro_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            return macro_f1
        elif average == 'micro':
            tp = np.sum((y_true_classes == y_pred_classes))
            fp = np.sum((y_true_classes != y_pred_classes))
            micro_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            micro_recall = tp / (tp + fp) if (tp + fp) > 0 else 0
            micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
            return micro_f1
        else:
            raise ValueError("average must be 'macro' or 'micro'")