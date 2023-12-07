import torch.nn as nn

class BinaryImageClassifier(nn.Module):
    """
    BinaryImageClassifier is a convolutional neural network (CNN) designed for
    binary image classification using PyTorch.

    Args:
        num_classes (int): Number of output classes for the final fully connected layer.
                           Default is 2.

    Attributes:
        conv1, conv2, conv3, conv4 (nn.Sequential): Convolutional layers with batch normalization
            and ReLU activation.
        maxpool (nn.MaxPool2d): Max pooling layer.
        fc1, fc2, fc3 (nn.Sequential): Fully connected layers with batch normalization and ReLU activation.
        dropout (nn.Dropout): Dropout layer to prevent overfitting.
    """
    def __init__(self, output_size=1):
        """
        Initializes the BinaryImageClassifier.

        Args:
            num_classes (int): Number of output classes for the final fully connected layer.
                               Default is 16.
        """
        # initialize the BinaryImageClassifier class, inheriting from nn.Module
        super(BinaryImageClassifier, self).__init__()

        # define convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # define pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # define fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(100352, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(512, output_size)

        # define dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Defines the forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor representing the image data.

        Returns:
            torch.Tensor: Output logits representing the class scores.
        """
        # forward pass through the network
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.maxpool(x)

        # flatten the input for the fully connected layers
        x = x.view(x.size(0), -1)

        # apply fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout(x)

        # final output layer
        logits = self.fc3(x)

        return logits
