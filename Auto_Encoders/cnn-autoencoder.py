import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

class CnnAutoencoder(nn.Module):
    def __init__(self):
        super(CnnAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # [16, 14, 14]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # [32, 7, 7]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # [64, 4, 4]
            nn.ReLU(),
            nn.Flatten(),                                           # Flatten to [1024]
            nn.Linear(64 * 4 * 4, 128),                            # Latent space [128]
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(128, 64 * 4 * 4),                           # Expand back to [64 * 4 * 4]
            nn.ReLU(),
            nn.Unflatten(1, (64, 4, 4)),                          # Reshape to [64, 4, 4]
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0), # [32, 7, 7]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # [16, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # [1, 28, 28]
            nn.Tanh()  # Output in range [-1, 1]
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed


class CnnAutoencoderWithClassifier(CnnAutoencoder):
    def __init__(self):
        super(CnnAutoencoderWithClassifier, self).__init__()
        self.classifier = nn.Linear(128, 10)  # 10 classes in FashionMNIST

    def forward(self, x):
        latent = self.encode(x)  # Latent features
        reconstructed = self.decode(latent)  # Reconstructed image
        classification = self.classifier(latent)  # Classification from latent space
        return reconstructed, classification

    def fit(self, train_loader, val_loader=None, num_epochs=10, lr=0.001, lambda_reconstruction=0.5):
        # Define optimizers and loss functions
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion_reconstruction = nn.MSELoss()  # Reconstruction loss
        criterion_classification = nn.CrossEntropyLoss()  # Classification loss

        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            running_loss = 0.0

            # Training loop
            for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                reconstructed, classification = self(images)

                # Compute the losses
                loss_reconstruction = criterion_reconstruction(reconstructed, images)
                loss_classification = criterion_classification(classification, labels)

                # Combine the losses
                loss = lambda_reconstruction * loss_reconstruction + (1 - lambda_reconstruction) * loss_classification

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Validation loop (optional)
            val_loss, val_reconstruction_loss, val_classification_loss, val_accuracy = self.evaluate(val_loader, lambda_reconstruction)

            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Training Loss: {running_loss/len(train_loader):.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Reconstruction Loss: {val_reconstruction_loss:.4f}, "
                  f"Val Classification Loss: {val_classification_loss:.4f}, "
                  f"Val Accuracy: {val_accuracy:.2f}%")

    def evaluate(self, data_loader, lambda_reconstruction):
        # Set the model to evaluation mode
        self.eval()
        val_loss = 0.0
        val_reconstruction_loss = 0.0
        val_classification_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data in data_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                reconstructed, classification = self(images)

                # Compute the losses
                loss_reconstruction = nn.MSELoss()(reconstructed, images)
                loss_classification = nn.CrossEntropyLoss()(classification, labels)

                # Combine the losses
                loss = lambda_reconstruction * loss_reconstruction + (1 - lambda_reconstruction) * loss_classification

                # Accumulate losses
                val_loss += loss.item()
                val_reconstruction_loss += loss_reconstruction.item()
                val_classification_loss += loss_classification.item()

                # Accuracy calculation
                _, predicted = torch.max(classification, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Average the losses and calculate accuracy
        val_loss /= len(data_loader)
        val_reconstruction_loss /= len(data_loader)
        val_classification_loss /= len(data_loader)
        accuracy = 100 * correct / total

        return val_loss, val_reconstruction_loss, val_classification_loss, accuracy

    def get_latent(self,X):
        with torch.no_grad():
            latent = self.encode(X)
        return latent.cpu().numpy()