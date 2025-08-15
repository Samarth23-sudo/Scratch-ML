import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class MultiTaskCNN(nn.Module):
    def __init__(self, task='classification', num_classes=10, num_conv_layers=2, dropout_rate=0.5):
        super(MultiTaskCNN, self).__init__()
        self.task = task
        self.dropout = dropout_rate
        self.num_conv_layers = num_conv_layers

        # Define convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1))  # Input channels = 1 (grayscale), output channels = 32
         # Additional layers based on num_conv_layers
        if self.num_conv_layers > 1:
            self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        if self.num_conv_layers > 2:
            self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1))

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 7x7 is the output size after two max-pooling operations for 28x28 input

        self.fc2_classification = nn.Linear(128, num_classes)  # For classification

        self.fc2_regression = nn.Linear(128, 1)  # For regression (output is a single value)

        self.droupout = nn.Dropout(p=self.dropout)

    def forward(self, x):


        # Pass through convolutional layers
        x = F.relu(self.conv1(x)) # Conv1 -> ReLU -> MaxPool
        x = self.pool(x)
        x = F.relu(self.conv2(x)) # Conv2 -> ReLU -> MaxPool
        x = self.pool(x)

        # Flatten the tensor for fully connected layers
        x = x.reshape(x.shape[0], -1)
        # Fully connected layer
        x = self.droupout(F.relu(self.fc1(x)))

        # Output layer depends on the task
        if self.task == 'classification':
            x = self.fc2_classification(x)  # Output for classification task
        elif self.task == 'regression':
            x = self.fc2_regression(x)  # Output for regression task

        return x


# Initialize the model, optimizer, and loss function
def train_model(model, train_loader, val_loader, task, num_epochs=10, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Choose loss function based on task
    if task == 'classification':
        criterion = nn.CrossEntropyLoss()  # For classification tasks
    elif task == 'regression':
        criterion = nn.MSELoss()  # For regression tasks

    # Lists to store losses for plotting or analysis
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Clear gradients

            # Forward pass
            outputs = model(images)
            if task == 'classification':
                loss = criterion(outputs, labels)
            elif task == 'regression':
                loss = criterion(outputs, labels.float())
            #print(loss)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                if task == 'classification':
                    loss = criterion(outputs, labels)
                elif task == 'regression':
                    loss = criterion(outputs, labels.float())
                running_val_loss += loss.item() * images.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    return train_losses, val_losses


def evaluate_model(model, test_loader, task,regression_tolerance=0.1):
    model.eval()
    test_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    criterion_classification = nn.CrossEntropyLoss()
    criterion_regression = nn.MSELoss()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # For classification, compute accuracy
            if task == 'classification':
                loss = criterion_classification(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)
            # For regression, compute MSE
            elif task == 'regression':
                loss = criterion_regression(outputs, labels)
                test_loss += loss.item() * images.size(0)
                
                # Calculate number of predictions within the tolerance
                tolerance = regression_tolerance * labels
                within_tolerance = torch.abs(outputs - labels) <= tolerance
                correct_predictions += within_tolerance.all(dim=1).sum().item()  # Counts per sample
                total_samples += labels.size(0)

    if task == 'classification':
        accuracy = 100 * correct_predictions / total_samples
        print(f'Classification Accuracy on test set: {accuracy:.2f}%')
    elif task == 'regression':
        mse = test_loss / len(test_loader.dataset)
        accuracy = 100 * correct_predictions / total_samples
        print(f'Regression MSE on test set: {mse:.4f}')
        print(f'Regression Accuracy on test set: {accuracy:.2f}%')



