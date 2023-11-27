import os
import glob
import torch
import pathlib
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from mpl_toolkits.mplot3d import Axes3D
from torch_geometric.datasets import ModelNet


class RandomJitterTransform(object):
    def __init__(self, sigma=0.01, clip=0.05):
        """
        Randomly jitter points in a point cloud.

        Args:
            sigma (float): Standard deviation of the jittering distribution.
            clip (float): Maximum absolute value to clip the jittered points.
        """
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data):
        """ 
        Randomly jitter points. Jittering is per point.

        Args:
            data (torch.Tensor): Data object containing the point clouds

        Returns:
            torch.Tensor: Jittered point clouds
        """
        # Get the number of points and the dimensionality of each point
        N, C = data.shape

        assert self.clip > 0

        # Generate jittered values from a normal distribution
        jittered_values = torch.FloatTensor(np.clip(self.sigma * np.random.randn(N, C), -1 * self.clip, self.clip))

        # Add the jittered values to the original points
        jittered_data = data + jittered_values

        return jittered_data


class RandomRotateTransform(object):
    def __init__(self):
        pass

    def __call__(self, data):
        """ 
        Randomly rotate the point clouds to augment the dataset.
        rotation is per shape based along ANY direction

        Args:
            data (torch.Tensor): Data object containing the point clouds

        Returns:
            torch.Tensor: Rotated point clouds
        """
        rotation_angle = np.random.uniform() * 2 * np.pi

        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)

        rotation_matrix = torch.FloatTensor([[cosval, 0, sinval], 
                                             [0, 1, 0], 
                                             [-sinval, 0, cosval]])

        rotated_data = torch.matmul(data, rotation_matrix)

        return rotated_data


class ScaleTransform(object):
    def __init__(self):
        pass

    def __call__(self, data):
        """ 
        Scaling transformation to make all points to be in the cube [0, 1]

        Args:
            data (torch.Tensor): Data object containing the point clouds

        Returns:
            torch.Tensor: Scaled point clouds
        """
        scaled_data = (data - data.min(dim=0).values) / (data.max(dim=0).values - data.min(dim=0).values)

        return scaled_data


class CustomModelNetDataset(Dataset):
    def __init__(self, data_root, transform, train):
        """
        Custom dataset for loading 3D point cloud data from ModelNet.

        Args:
            data_root (str): Root directory of the dataset.
            transform (callable): Optional transform to be applied on a sample.
            train (bool): Flag indicating whether to load training or testing data.
        """
        super(CustomModelNetDataset, self).__init__()
        self.data_root = data_root
        self.transform = transform
        self.train = train
        self.file_paths = []
        self.class_to_idx = {}
        self.find_classes()
        self.get_file_list()

    def find_classes(self):
        """
        Finds and assigns numerical indices to class labels based on subdirectories in the dataset.

        Raises:
            FileNotFoundError: If no classes are found in the specified data root.
        """
        classes = sorted(entry.name for entry in os.scandir(self.data_root) if entry.is_dir())

        if not classes:
            raise FileNotFoundError(f"Couldn't find any classes in {self.data_root}... please check file structure.")
        
        self.class_to_idx = {class_name: i for i, class_name in enumerate(classes)}

    def get_file_list(self):
        """
        Retrieves the list of file paths for either training or testing data.

        Sets:
            self.file_paths (list): List of file paths for 3D point cloud data.
        """
        if self.train:
            self.file_paths = list(pathlib.Path(self.data_root).glob("*/train/*.off"))
        else:
            self.file_paths = list(pathlib.Path(self.data_root).glob("*/test/*.off"))

    def load_point_cloud(self, file_path, skiprows=2, maxrows=12636, type=np.float32):
        """
        Loads 3D point cloud data from an OFF file.

        Args:
            file_path (str): Path to the OFF file.
            skiprows (int, optional): Number of header lines to skip. Defaults to 2.
            maxrows (int, optional): Maximum number of rows to read. Defaults to 12636.
            type (numpy.dtype, optional): Data type to be used. Defaults to np.float32.

        Returns:
            torch.Tensor: 3D point cloud data as a PyTorch tensor.
        """
        with open(file_path, "r") as file:
            lines = file.readlines()[skiprows:]

        # Extract x, y, z coordinates
        coordinates = [list(map(float, line.strip().split()[:3])) for line in lines]

        np_array = np.array(coordinates, dtype=type)

        return torch.from_numpy(np_array)

    def __getitem__(self, index):
        """
        Retrieves a specific item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: Tuple containing the 3D point cloud data and its corresponding class index.
        """
        pntcld_path = self.file_paths[index]

        class_name = pntcld_path.parent.parent.name

        point_cloud = self.load_point_cloud(file_path=pntcld_path)

        if self.transform is not None:
            point_cloud = self.transform(point_cloud)
        
        return point_cloud, self.class_to_idx[class_name]

    def __len__(self):
        """
        Returns the total number of items in the dataset.

        Returns:
            int: Total number of items in the dataset.
        """
        return len(self.file_paths)


class PointNet(nn.Module):
    def __init__(self, num_classes=10):
        super(PointNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Apply your network operations
        x = F.relu(self.conv1(x.permute(0, 2, 1)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc1(x.mean(dim=-1)))
        x = self.fc2(x)

        return x


class EarlyStopping:
    """ 
    Early stops the training if validation loss doesn't improve after a given patience.
    Copyright (c) 2018 Bjarte Mehus Sunde
    """

    def __init__(self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print):
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
            self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class Solver(object):
    def __init__(self, epochs, trainloader, testloader, device, model, optimizer, criterion, patience):
        """
        A class to handle training and validation of a PyTorch neural network.

        Args:
            epochs (int): Number of training epochs.
            trainloader (torch.utils.data.DataLoader): DataLoader for training data.
            testloader (torch.utils.data.DataLoader): DataLoader for validation data.
            device (torch.device): Device on which to perform training and validation.
            model (torch.nn.Module): Neural network model.
            optimizer (torch.optim.Optimizer): Optimization algorithm.
            criterion (torch.nn.Module): Loss function.
            patience (int): Number of epochs to wait for improvement before early stopping.
        """
        self.epochs = epochs
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.patience = patience

    def train_net(self):
        """
        Trains the neural network using the specified DataLoader for training data.

        This method also performs early stopping based on the validation loss.

        Prints training and validation statistics for each epoch.
        """
        # Lists to track training and validation losses
        train_losses = []
        valid_losses = []
        # Lists to track average losses per epoch
        avg_train_losses = []
        avg_valid_losses = []

        # Initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)

        for epoch in range(self.epochs):
            self.model.train()

            # Use tqdm for a progress bar during training
            loop = tqdm(iterable=enumerate(self.trainloader),
                        total=len(self.trainloader),
                        leave=False)

            for _, (x_train, y_train) in loop:
                # Move data and labels to the specified device
                x_train = x_train.to(self.device)
                y_train = y_train.to(self.device)

                # Forward pass: compute predicted outputs by passing inputs to the model
                y_pred = self.model(x_train)

                # Calculate the loss
                loss = self.criterion(y_pred, y_train)

                # Clear the gradients of all optimized variables
                self.optimizer.zero_grad()

                # Backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                # Perform a single optimization step (parameter update)
                self.optimizer.step()

                # Record training loss
                train_losses.append(loss.item())

            # Validate the model on the validation set
            self.valid_net(valid_losses=valid_losses)

            # Print training/validation statistics
            # Calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            epoch_len = len(str(self.epochs))

            print(f"[{epoch:>{epoch_len}}/{self.epochs:>{epoch_len}}] "
                  f"train_loss: {train_loss:.5f} "
                  f"valid_loss: {valid_loss:.5f}")

            # Clear lists to track next epoch
            train_losses = []
            valid_losses = []

            # Early stopping checks for improvement in validation loss
            early_stopping(valid_loss, self.model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

    def valid_net(self, valid_losses):
        """
        Validates the neural network on the specified DataLoader for validation data.

        Records validation losses in the provided list.

        Args:
            valid_losses (list): List to record validation losses.
        """
        self.model.eval()

        # Use tqdm for a progress bar during validation
        with torch.inference_mode():
            loop = tqdm(iterable=enumerate(self.testloader),
                        total=len(self.testloader),
                        leave=False)

            for _, (x_valid, y_valid) in enumerate(loop):
                x_valid = x_valid.to(self.device)
                y_valid = y_valid.to(self.device)

                # Forward pass: compute predicted outputs by passing inputs to the model
                y_pred = self.model(x_valid)

                # Calculate the loss
                loss = self.criterion(y_pred, y_valid)

                # Record validation loss
                valid_losses.append(loss.item())

        # Set the model back to training mode
        self.model.train()


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

def visualize_pointcloud(point_cloud):
    """
    Visualizes a 3D point cloud using a scatter plot.

    Args:
        point_cloud (torch.Tensor): 3D point cloud data as a PyTorch tensor with shape (N, 3),
                                    where N is the number of points, and each point has X, Y, Z coordinates.
    """
    # Create a new 3D figure
    fig = plt.figure()
    
    # Add a 3D subplot to the figure
    ax = fig.add_subplot(111, projection="3d")
    
    # Extract X, Y, Z coordinates from the point cloud
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    # Scatter plot of the points with red color and spherical markers
    ax.scatter(x, y, z, c='r', marker='o')

    # Set labels for the axes
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Show the plot
    plt.show()


if __name__ == "__main__":

    batch_size = 32
    base_path = "./point-clouds/data/raw/"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_transform = transforms.Compose([RandomJitterTransform(),
                                          RandomJitterTransform(),
                                          ScaleTransform()])

    test_transform = transforms.Compose([ScaleTransform()])


    train_dataset = CustomModelNetDataset(data_root=base_path, transform=train_transform, train=True)    
    test_dataset = CustomModelNetDataset(data_root=base_path, transform=test_transform, train=False)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)    
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)


    point_net = PointNet().to(device)
    optimizer = torch.optim.Adam(params=point_net.parameters(), lr=0.001, betas=(0.9, 0.999))
    loss_fn = nn.CrossEntropyLoss()

    solver = Solver(epochs=3, 
                    trainloader=train_loader,
                    testloader=test_loader,
                    device=device,
                    model=point_net,
                    optimizer=optimizer,
                    criterion=loss_fn,
                    patience=5)

    solver.train_net()


    # for i, (x_train, y_train) in enumerate(train_loader):
    #     print(i)
    #     print(x_train)
    #     print(y_train)
    #     a = input("...")





    # tensor_array = torch.from_numpy(np.loadtxt("./point-clouds/data/raw/table/train/table_0001.off", 
    #                                            skiprows=2, max_rows=12636, dtype=np.float32))

    # tensor_array = tensor_array.unsqueeze(0)

    # print(tensor_array.permute(0, 2, 1))

    # test_pointcloud()











# def get_modelnet_10(datadir, batch_size):
#     train_transform = transforms.Compose(
#         [
#             RandomJitterTransform(),
#             RandomJitterTransform(),
#             ScaleTransform(),
#         ]
#     )

#     valid_transform = transforms.Compose(
#         [
#             ScaleTransform(),
#         ]
#     )

#     train_data = ModelNet(
#         root=datadir, name="10", train=True, transform=train_transform
#     )
#     valid_data = ModelNet(
#         root=datadir, name="10", train=False, transform=valid_transform
#     )

#     # Use the custom collate function in your DataLoader
#     trainloader = DataLoader(
#         train_data, batch_size=batch_size, shuffle=True)
#     testloader = DataLoader(
#         valid_data, batch_size=batch_size, shuffle=False)

#     return trainloader, testloader
