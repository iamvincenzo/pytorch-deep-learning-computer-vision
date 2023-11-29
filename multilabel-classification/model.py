import torch.nn as nn

class Model(nn.Module):
    """
    PointNet for classification using T-Net for input and feature transformations.

    Args:
        point_dimension (int): Dimensionality of input point cloud.
        num_classes (int): Number of output classes.
    """
    def __init__(self, num_classes=10):
        super(Model, self).__init__()

        # define the number of output classes
        self.num_classes = num_classes

        # fully connected layers for classification
        self.fc1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(64, self.num_classes)

    def forward(self, x):
        # fully connected layers and dropout for classification
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout(x)

        # output logits
        logits = self.fc3(x)        

        return logits
