import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import io
import json

class Linear_Regression():

    # Initiating the parameters (learning rate & no. of iterations)
    def __init__(self, learning_rate=0.01, no_of_iterations=1000, degree=1, regularization=None, lambda_=0.1):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.degree = degree
        self.iteration_predictions = []
        #regularisation
        self.regularization = regularization
        self.lambda_ = lambda_

    # fiting the model to the training data
    def fit(self, X, Y):        
        # Convert X and Y to NumPy arrays if they aren't already
        self.X = np.array(X).reshape(-1, 1)  # Ensure X is 2D
        self.Y = np.array(Y)
        
        # Expanding X to include polynomial features
        self.X_poly = self._expand_features(self.X)
        # no. of training examples & number of features
        self.m, self.n = self.X_poly.shape  # number of rows & columns

        # Initiating the weights and bias 
        self.w = np.zeros(self.n)
        self.b = 0
        # self.X = X
        # self.Y = Y

        # Implementing Gradient Descent
        for i in range(self.no_of_iterations):
            self.update_weights(i)
            
        # self.create_gif()

    # Function to update weights and bias using gradient descent
    def update_weights(self,iteration):
        Y_prediction = self.predict(self.X)

        # Store prediction for this iteration
        if iteration % 5 == 0:  # Store only every 10th iteration to reduce memory usage
            self.iteration_predictions.append(Y_prediction.copy())
            
        # calculate gradients and go towards a smaller cost
        dw = - (2 * (self.X_poly.T).dot(self.Y - Y_prediction)) / self.m     #cost function is averge mse(Y_pred = w.X+b) and avg_mse=(Y-Y_pred)^2/m
        db = - 2 * np.sum(self.Y - Y_prediction) / self.m
        
        # Add regularization term if needed
        if self.regularization == 'L1':
            dw += self.lambda_ * np.sign(self.w)
        elif self.regularization == 'L2':
            dw += 2*self.lambda_ * self.w

        # Updating the weights and bias
        self.w = self.w - self.learning_rate * dw  #if dw is positive then it means the slope is postive and on 
        self.b = self.b - self.learning_rate * db  #decreasing the value of w, the total cost decreases(thus -dw*learning_Rate) ensures that we move in the direction of decreasing cost
        
    # Function to make predictions
    def predict(self, X):
        X_poly = self._expand_features(X)
        return X_poly.dot(self.w) + self.b
    
     #to expand features for polynomial regression
    def _expand_features(self, X):
        # Expand the features to polynomial terms
        X_poly = np.ones((X.shape[0], self.degree + 1))  # +1 for the bias term(here we have colums=degree+1(for bias) and number of rows is same as X)
        for i in range(1, self.degree + 1):
            X_poly[:, i] = X[:, 0] ** i  #for all rows in the ith column raise all the values of ith column by i(to create a polynomial)
        return X_poly
    
     # Method to calculate Mean Squared Error
    def calculate_mse(self, Y_true, Y_pred):
        mse = np.mean((Y_true - Y_pred) ** 2)
        return mse
    
    def calculate_std_dev(self, Y_true, Y_pred):
        std_dev = np.std(Y_true - Y_pred)
        return std_dev
    
    def calculate_variance(self, Y_true,Y_pred):
        var_array = np.abs(Y_true-Y_pred)
        variance = np.var(var_array)
        return variance
    
    def shuffle_and_split(self,X, Y):
        # Combine X and Y, shuffle, and split
        X = np.array(X).reshape(-1, 1)
        Y = np.array(Y).reshape(-1, 1)
        data = np.hstack((X, Y))
        np.random.shuffle(data)
        
        # Split the shuffled data
        train_size = int(0.8 * len(data))
        val_size = int(0.1 * len(data))
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        X_train, Y_train = train_data[:, :-1], train_data[:, -1]
        X_val, Y_val = val_data[:, :-1], val_data[:, -1]
        X_test, Y_test = test_data[:, :-1], test_data[:, -1]
        
        return X_train, Y_train, X_val, Y_val, X_test, Y_test
    
    

    #to save the model parameters
    def save_model(self, filename='best_model.json'):
        model_params = {
            'weights': self.w.tolist(),
            'bias': self.b,
            'degree': self.degree
        }
        with open(filename, 'w') as f:
            json.dump(model_params, f)
        
    # to load the model parameters
    def load_model(self, filename='best_model.json'):
        with open(filename, 'r') as f:
            model_params = json.load(f)
        self.w = np.array(model_params['weights'])
        self.b = model_params['bias']
        self.degree = model_params['degree']
        
    def create_iteration_gif(self, X, Y, filename='polynomial_regression_fit.gif'):
        frames = []
        for i, Y_pred in enumerate(self.iteration_predictions):
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))

            # Plot the training data and the fitted line
            axs[0, 0].scatter(X, Y, color='blue', label='Training data')
            axs[0, 0].plot(np.sort(X, axis=0), Y_pred[np.argsort(X, axis=0)], color='red', label='Fitted line')
            axs[0, 0].set_title(f"Iteration {i * 5 + 1}")
            axs[0, 0].set_xlabel("X")
            axs[0, 0].set_ylabel("Y")
            axs[0, 0].legend()

            # Calculate and display metrics
            mse = self.calculate_mse(Y, Y_pred)
            std_dev = self.calculate_std_dev(Y, Y_pred)
            variance = self.calculate_variance(Y, Y_pred)
            
            # Display metrics in other subplots
            axs[0, 1].text(0.5, 0.5, f"MSE: {mse:.4f}", fontsize=12, ha='center')
            axs[0, 1].set_axis_off()
            axs[1, 0].text(0.5, 0.5, f"Standard Deviation: {std_dev:.4f}", fontsize=12, ha='center')
            axs[1, 0].set_axis_off()
            axs[1, 1].text(0.5, 0.5, f"Variance: {variance:.4f}", fontsize=12, ha='center')
            axs[1, 1].set_axis_off()

            # Save the plot to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            frames.append(imageio.imread(buf))
            plt.close(fig)
        
        # Save the frames as a GIF
        imageio.mimsave(filename, frames, fps=10)