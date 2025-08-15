import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


class MultiMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
        label = torch.tensor(label, dtype=torch.float32)
        if self.transform:
            img = self.transform(img)
        return img, label

class MultiLabelCNN(nn.Module):
    def __init__(self, num_conv_layers=3, dropout_rate=0.5, max_digits=5, num_classes=10):
        super(MultiLabelCNN, self).__init__()

        self.num_conv_layers = num_conv_layers
        self.dropout_rate = dropout_rate
        self.max_digits = max_digits
        self.num_classes = num_classes

        # Define convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)

        # Fully connected layer
        if(self.num_conv_layers==2):
            self.fc = nn.Linear(64 * 7 * 7, num_classes * max_digits)
        else:
            self.fc = nn.Linear(128 * 3 * 3, num_classes * max_digits)

        # Dropout
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # First conv layer with relu and maxpool
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # Second conv layer if num_conv_layers > 1
        if self.num_conv_layers > 1:
            x = F.relu(self.conv2(x))
            x = self.pool(x)
        # Third conv layer if num_conv_layers > 2
        if self.num_conv_layers > 2:
            x = F.relu(self.conv3(x))
            x = self.pool(x)
        # Flatten the tensor for fully connected layer
        x = x.view(x.size(0), -1)
        # Fully connected layer
        x = self.dropout(F.relu(self.fc(x)))

        # Reshape output to [batch_size, num_classes, max_digits]
        return torch.sigmoid(x).view(-1, self.num_classes, self.max_digits)



#Data Loading and Preprocessing(chnage according to need of task)
def load_mnist_data(data_path, max_digits=5):
    train_data, val_data, test_data = [], [], []
    train_labels, val_labels, test_labels = [], [], []

    for split in ['train', 'val', 'test']:
        split_path = os.path.join(data_path, split)
        for label_name in os.listdir(split_path):
            label = [int(digit) for digit in label_name]

            # Pad the label array to max_digits and create a one-hot encoding for each digit position
            padded_label = label + [0] * (max_digits - len(label))
            one_hot_label = np.zeros((10, max_digits), dtype=np.float32)
            for idx, digit in enumerate(padded_label):
                one_hot_label[digit, idx] = 1.0
            image_folder = os.path.join(split_path, label_name)
            for img_file in os.listdir(image_folder):
                img_path = os.path.join(image_folder, img_file)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (28, 28))

                if split == 'train':
                    train_data.append(image)
                    train_labels.append(one_hot_label)
                elif split == 'val':
                    val_data.append(image)
                    val_labels.append(one_hot_label)
                elif split == 'test':
                    test_data.append(image)
                    test_labels.append(one_hot_label)

    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)

data_path = 'double_mnist'
(train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_mnist_data(data_path)

train_dataset = MultiMNISTDataset(train_data, train_labels)
val_dataset = MultiMNISTDataset(val_data, val_labels)
test_dataset = MultiMNISTDataset(test_data, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
