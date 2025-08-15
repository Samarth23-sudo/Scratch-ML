import numpy as np
from MLP_regression import MLPRegressor

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    print(y_true,y_pred)
    return 2*(y_pred - y_true)

class AutoEncoder:
    def __init__(self, input_size, latent_dim, hidden_layers, learning_rate=0.01, activation='sigmoid', optimizer='sgd', batch_size=100, epochs=100):
        """
        Initializes the AutoEncoder with an encoder and decoder.
        
        Args:
            input_size (int): Number of features in the input vector.
            latent_dim (int): The number of dimensions to reduce to (latent space).
            hidden_layers (list): List of neurons in hidden layers for both encoder and decoder.
            learning_rate (float): Learning rate for optimization.
            activation (str): Activation function to use.
            optimizer (str): Optimizer type (e.g., 'sgd').
            batch_size (int): Batch size for training.
            epochs (int): Number of epochs for training.
        """
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Encoder: MLP that reduces the input to the latent dimension
        self.encoder = MLPRegressor(input_size=input_size, hidden_layers=hidden_layers, output_size=latent_dim, 
                                    learning_rate=learning_rate, activation=activation, optimizer=optimizer, 
                                    batch_size=batch_size, epochs=epochs)
        
        # Decoder: MLP that reconstructs the input from the latent dimension
        self.decoder = MLPRegressor(input_size=latent_dim, hidden_layers=hidden_layers[::-1], output_size=input_size, 
                                    learning_rate=learning_rate, activation=activation, optimizer=optimizer, 
                                    batch_size=batch_size, epochs=epochs)
    
    def fit(self, X):
        """
        Trains the autoencoder using forward and backward propagation.

        Args:
            X (np.array): Input dataset to train on (shape: [num_samples, input_size]).
        """
        for epoch in range(self.epochs):
            epoch_loss = 0
            for i in range(0, X.shape[0], X.shape[0]):
                # Get a mini-batch of data
                X_batch = X[i:i + self.batch_size]

                # Encoder forward pass: Get latent space
                latent_space_activation, latent_space_preactivation = self.encoder._forward(X_batch)
                
                # Decoder forward pass: Reconstruct the input from latent space
                reconstructed_X_activation, reconstructed_X_preactivation = self.decoder._forward(latent_space_activation[-1])  # latent_space[-1] is the final output from the encoder

                # Calculate MSE loss between reconstructed input and original input
                #print(reconstructed_X_activation[-1])
                loss = mse_loss(X_batch, reconstructed_X_activation[-1])  # reconstructed_X[-1] is the final output from the decoder
                epoch_loss = loss

                # Compute gradients for the decoder
                gradients_decoder_w, gradients_decoder_b = self.decoder._backward(latent_space_activation[-1], X_batch, reconstructed_X_activation, reconstructed_X_preactivation)
                
                # Backpropagate the error from the decoder into the encoder
                gradients_encoder_w, gradients_encoder_b = self.encoder._backward(X_batch, latent_space_activation[-1], latent_space_activation, latent_space_preactivation)

                # Update weights of decoder and encoder
                for j in range(len(self.decoder.weights)):
                    self.decoder.weights[j] -= self.learning_rate * gradients_decoder_w[j]
                    self.decoder.biases[j] -= self.learning_rate * gradients_decoder_b[j]
                    
                # Update weights of the encoder
                for j in range(len(self.encoder.weights)):
                    self.encoder.weights[j] -= self.learning_rate * gradients_encoder_w[j]
                    self.encoder.biases[j] -= self.learning_rate * gradients_encoder_b[j]

            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss}")


            
    def get_latent(self, X):
        """
        Encodes the input dataset and returns the latent space (compressed representation).
        
        Args:
            X (np.array): Input dataset to be compressed (shape: [num_samples, input_size]).
            
        Returns:
            np.array: Compressed latent representation (shape: [num_samples, latent_dim]).
        """
        latent_space = self.encoder.predict(X)  # Get latent representation from the encoder
        return latent_space
