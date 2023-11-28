# reference: https://datascienceub.medium.com/pointnet-implementation-explained-visually-c7e300139698

import torch
import torch.nn as nn

class TNet(nn.Module):
    """
    T-Net aligns all input sets to a canonical space by learning a transformation matrix.

    Args:
        input_dim (int): Dimensionality of input features.
        output_dim (int): Dimensionality of the output features after transformation.
    """
    def __init__(self, input_dim=3, output_dim=3):
        super(TNet, self).__init__()

        # define the output dimension for the T-Net transformation
        self.output_dim = output_dim

        # shared MLP layers for feature extraction
        self.sharedMLP1 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.sharedMLP2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.sharedMLP3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        # fully connected layers for final transformation matrix
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fc3 = nn.Linear(
            in_features=128, out_features=output_dim * output_dim
        )

    def forward(self, x):
        # permute input for proper shape and apply shared MLP layers
        x = x.permute(0, 2, 1)
        x = self.sharedMLP1(x)
        x = self.sharedMLP2(x)
        x = self.sharedMLP3(x)
        
        # global max pooling
        # transform the entire point cloud in a single vector that not depend on the 
        # order of the points or on the number of points
        x, _ = torch.max(x, 2)

        # apply fully connected layers to obtain the transformation matrix
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        # add an identity matrix to the transformation matrix
        iden = torch.eye(self.output_dim)
        if torch.cuda.is_available():
            iden = iden.cuda()
        x = x.view(-1, self.output_dim, self.output_dim) + iden

        return x

class PointNetClassif(nn.Module):
    """
    PointNet for classification using T-Net for input and feature transformations.

    Args:
        point_dimension (int): Dimensionality of input point cloud.
        num_classes (int): Number of output classes.
    """
    def __init__(self, point_dimension=3, num_classes=10):
        super(PointNetClassif, self).__init__()

        # define the number of output classes
        self.num_classes = num_classes

        # T-Net for input transformation and feature transformation
        self.input_transform = TNet(input_dim=point_dimension, output_dim=point_dimension)
        self.feature_transform = TNet(input_dim=64, output_dim=64)

        # shared MLP layers for feature extraction
        self.sharedMLP1 = nn.Sequential(
            nn.Conv1d(in_channels=point_dimension, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.sharedMLP2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.sharedMLP3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.sharedMLP4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.sharedMLP5 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

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
        # apply T-Net for input transformation
        input_t = self.input_transform(x)
        x = torch.bmm(x, input_t)
        x = x.permute(0, 2, 1)

        # apply shared MLP layers for feature extraction
        x = self.sharedMLP1(x)
        x = self.sharedMLP2(x)
        x = x.permute(0, 2, 1)

        # apply T-Net for feature transformation
        feature_t = self.feature_transform(x)
        x = torch.bmm(x, feature_t)
        x = x.permute(0, 2, 1)

        # continue with shared MLP layers
        x = self.sharedMLP3(x)
        x = self.sharedMLP4(x)
        x = self.sharedMLP5(x)

        # global max pooling
        # transform the entire point cloud in a single vector that not depend on the 
        # order of the points or on the number of points
        x, _ = torch.max(x, 2)

        # fully connected layers and dropout for classification
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout(x)

        # output logits
        logits = self.fc3(x)        

        return logits, feature_t
