from typing import Any
import torch
import numpy as np
import torch_geometric
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from mpl_toolkits.mplot3d import Axes3D
from torch_geometric.datasets import ModelNet


def test_pointcloud():
    # sample 3D point cloud with three features (x, y, z)
    # batch size: 1, Number of points: 5, Number of features: 3
    point_cloud = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0],
            ]
        ]
    )

    torch.manual_seed(42)

    print(f"Point-cloud shape before: {point_cloud.shape}")

    print(point_cloud)

    # reshape the input to (batch_size, features, n-points) for Conv1D
    point_cloud = point_cloud.permute(0, 2, 1)

    print(f"Point-cloud shape after: {point_cloud.shape}")

    # define a simple 1D convolutional layer
    conv1d_layer = nn.Conv1d(in_channels=3, out_channels=2, kernel_size=1)

    # apply the convolution operation
    output = conv1d_layer(point_cloud)

    # print the input, convolution operation, and output
    print("Input (3D point cloud):")
    print(point_cloud)

    print("\nConv1D Layer:")
    print(conv1d_layer.weight.data)
    print(conv1d_layer.bias.data)

    print("\nOutput after Conv1D:")
    print(output)

    # print manual calculations for the first output values
    print((1 * 0.4414) + (2 * 0.4792) + (3 * -0.1353) - 0.2811)  # 1st kernel
    print((4 * 0.4414) + (5 * 0.4792) + (6 * -0.1353) - 0.2811)  # 1st kernel
    print((7 * 0.4414) + (8 * 0.4792) + (9 * -0.1353) - 0.2811)  # 1st kernel
    print((13 * 0.5304) + (14 * -0.1265) + (15 * 0.1165) + 0.3391)  # 2nd kernel


class RandomJitterTransform(object):
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data):
        """Randomly jitter points. jittering is per point.

        Args:
            data (torch_geometric.data.Data): Data object containing the point clouds

        Returns:
            torch_geometric.data.Data: Jittered point clouds
        """
        jittered_data = data.clone()

        # Assuming your points are stored in the 'pos' attribute
        jittered_data.pos += torch.FloatTensor(
            np.clip(
                self.sigma * np.random.randn(data.num_nodes, 3),
                -1 * self.clip,
                self.clip,
            )
        )

        return jittered_data


class RandomRotateTransform(object):
    def __init__(self):
        pass

    def __call__(self, data):
        """Randomly rotate the point clouds to augment the dataset.
            rotation is per shape based along ANY direction

        Args:
            data (torch_geometric.data.Data): Data object containing the point clouds

        Returns:
            torch_geometric.data.Data: Rotated point clouds
        """
        rotation_angle = np.random.uniform() * 2 * np.pi

        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)

        rotation_matrix = torch.FloatTensor(
            [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
        )

        # Assuming your points are stored in the 'pos' attribute
        data.pos = torch.matmul(data.pos, rotation_matrix)

        return data


class ScaleTransform(object):
    def __init__(self):
        pass

    def __call__(self, data):
        """Scaling transformation to make all points to be in the cube [0, 1]

        Args:
            data (torch_geometric.data.Data): Data object containing the point clouds

        Returns:
            torch_geometric.data.Data: Scaled point clouds
        """
        # Assuming your points are stored in the 'pos' attribute
        data.pos = (data.pos - data.pos.min(dim=0).values) / (
            data.pos.max(dim=0).values - data.pos.min(dim=0).values
        )

        return data


def get_modelnet_10(datadir, batch_size):
    train_transform = transforms.Compose(
        [
            RandomJitterTransform(),
            RandomJitterTransform(),
            ScaleTransform(),
        ]
    )

    valid_transform = transforms.Compose(
        [
            ScaleTransform(),
        ]
    )

    train_data = ModelNet(
        root=datadir, name="10", train=True, transform=train_transform
    )
    valid_data = ModelNet(
        root=datadir, name="10", train=False, transform=valid_transform
    )

    # Use the custom collate function in your DataLoader
    trainloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(
        valid_data, batch_size=batch_size, shuffle=False)

    return trainloader, validloader


class PointNet(nn.Module):
    def __init__(self, num_classes=10):
        super(PointNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, data):
        # Assuming 'data' is a Batch object with 'pos' attribute
        x = data.pos  # Get the point positions

        # Apply your network operations
        x = F.relu(self.conv1(x.permute(0, 2, 1).to_dense()))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc1(x.mean(dim=-1)))
        x = self.fc2(x)

        return x


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    Copyright (c) 2018 Bjarte Mehus Sunde"""

    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class Solver(object):
    def __init__(
        self,
        epochs,
        trainloader,
        validloader,
        device,
        model,
        optimizer,
        criterion,
        patience,
    ):
        self.epochs = epochs
        self.trainloader = trainloader
        self.validloader = validloader
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.patience = patience

    def train_net(self):
        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)

        for epoch in range(self.epochs):
            self.model.train()

            loop = tqdm(
                iterable=enumerate(self.trainloader),
                total=len(self.trainloader),
                leave=False,
            )

            for _, (x_train, y_train) in loop:
                # put data and labels into the correct device
                x_train = x_train.to(self.device)
                y_train = y_train.to(self.device)

                # forward pass: compute predicted outputs by passing inputs to the model
                y_pred = self.model(x_train)

                # calculate the loss
                loss = self.criterion(y_pred, y_train)

                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()

                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                # perform a single optimization step (parameter update)
                self.optimizer.step()

                # record training loss
                train_losses.append(loss.item())

            self.valid_net(valid_losses=valid_losses)

            # print training/validation statistics
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            epoch_len = len(str(self.epochs))

            print(
                f"[{epoch:>{epoch_len}}/{self.epochs:>{epoch_len}}] "
                f"train_loss: {train_loss:.5f} "
                f"valid_loss: {valid_loss:.5f}"
            )

            # clear lists to track next epoch
            train_losses = []
            valid_losses = []

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, self.model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

    def valid_net(self, valid_losses):
        self.model.eval()

        with torch.inference_mode():
            loop = tqdm(
                iterable=enumerate(self.validloader),
                total=len(self.validloader),
                leave=False,
            )

            for _, (x_valid, y_valid) in enumerate(loop):
                x_valid = x_valid.to(self.device)
                y_valid = y_valid.to(self.device)

                # forward pass: compute predicted outputs by passing inputs to the model
                y_pred = self.model(x_valid)

                # calculate the loss
                loss = self.criterion(y_pred, y_valid)

                # record validation loss
                valid_losses.append(loss.item())

        self.model.train()


def visualize_pointcloud(point_cloud):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    ax.scatter(x, y, z, c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


if __name__ == "__main__":

    trainloader, validloader = get_modelnet_10(datadir="./point-clouds/data", batch_size=32)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    point_net = PointNet().to(device)
    optimizer = torch.optim.Adam(params=point_net.parameters(), lr=0.001, betas=(0.9, 0.999))
    loss_fn = nn.CrossEntropyLoss()

    solver = Solver(epochs=3, 
                    trainloader=trainloader,
                    validloader=validloader,
                    device=device,
                    model=point_net,
                    optimizer=optimizer,
                    criterion=loss_fn,
                    patience=5)

    solver.train_net()

    # test_pointcloud()
