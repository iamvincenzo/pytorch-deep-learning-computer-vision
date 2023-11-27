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
from torch.nn.utils.rnn import pad_sequence


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
        # get the number of points and the dimensionality of each point
        N, C = data.shape

        assert self.clip > 0

        # generate jittered values from a normal distribution
        jittered_values = torch.FloatTensor(np.clip(self.sigma * np.random.randn(N, C), -1 * self.clip, self.clip))

        # add the jittered values to the original points
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
        # generate a random rotation angle
        rotation_angle = np.random.uniform() * 2 * np.pi

        # compute cosine and sine values for the rotation angle
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)

        # create a 3x3 rotation matrix for the rotation angle
        rotation_matrix = torch.FloatTensor([[cosval, 0, sinval], 
                                             [0, 1, 0], 
                                             [-sinval, 0, cosval]])

        # apply the rotation matrix to the input point clouds
        rotated_data = torch.matmul(data, rotation_matrix)

        # return the rotated point clouds
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
        # calculate the minimum and maximum values along each dimension
        min_values = data.min(dim=0).values
        max_values = data.max(dim=0).values

        # scale the data to fit within the cube [0, 1]
        scaled_data = (data - min_values) / (max_values - min_values)

        # return the scaled point clouds
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

    def load_point_cloud(self, file_path, skiprows=2, n_coord=3, type=np.float32):
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

            # extract x, y, z coordinates
            coordinates = [list(map(float, line.strip().split()))
                           for line in lines if len(line.strip().split()) == n_coord]
            
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
        # apply your network operations
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
        print(f"\nStarting training...")

        # lists to track training and validation losses
        train_losses = []
        valid_losses = []
        # lists to track average losses per epoch
        avg_train_losses = []
        avg_valid_losses = []

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)
        
        self.model.train()

        for epoch in range(self.epochs):
            print(f"\nTraining iteration | Epoch[{epoch + 1}/{self.epochs}]")

            predictions_list = []
            targets_list = []

            # use tqdm for a progress bar during training
            loop = tqdm(iterable=enumerate(self.trainloader),
                        total=len(self.trainloader),
                        leave=True)

            for _, (x_train, y_train) in loop:
                # move data and labels to the specified device
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

                # record predictions and true labels
                predictions_list.append(y_pred)
                targets_list.append(y_train)

            all_preds = torch.cat(predictions_list, dim=0)
            all_targets = torch.cat(targets_list, dim=0)
            train_accuracy = self.compute_accuracy(logits=all_preds,
                                                   target=all_targets)

            # validate the model on the validation set
            valid_accuracy = 0
            self.valid_net(valid_losses=valid_losses, 
                           valid_accuracy=valid_accuracy)

            # print training/validation statistics
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            # print some statistics
            print(f"\nEpoch[{epoch + 1}/{self.epochs}] | train-loss: {train_loss:.4f} |"
                  f"validation-loss: {valid_loss:.4f} | train-accuracy: {train_accuracy} |"
                  f"valid-accuracy: {valid_accuracy}")

            # clear lists to track next epoch
            train_losses = []
            valid_losses = []

            # early stopping checks for improvement in validation loss
            early_stopping(valid_loss, self.model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        print("\nTraining model Done...\n")

    def valid_net(self, valid_losses, valid_accuracy):
        """
        Validates the neural network on the specified DataLoader for validation data.

        Records validation losses in the provided list.

        Args:
            valid_losses (list): List to record validation losses.
        """
        print(f"\nStarting validation...\n")

        self.model.eval()

        predictions_list = []
        targets_list = []

        # use tqdm for a progress bar during validation
        with torch.inference_mode():
            loop = tqdm(iterable=enumerate(self.testloader), 
                        total=len(self.testloader), 
                        leave=True)

            for _, (x_valid, y_valid) in loop:
                # move data and labels to the specified device
                x_valid = x_valid.to(self.device)
                y_valid = y_valid.to(self.device)

                # forward pass: compute predicted outputs by passing inputs to the model
                y_pred = self.model(x_valid)

                # calculate the loss
                loss = self.criterion(y_pred, y_valid)

                # record validation loss
                valid_losses.append(loss.item())

                # record predictions and true labels
                predictions_list.append(y_pred)
                targets_list.append(y_valid)

            all_preds = torch.cat(predictions_list, dim=0)
            all_targets = torch.cat(targets_list, dim=0)
            valid_accuracy = self.compute_accuracy(logits=all_preds, 
                                                   target=all_targets)
            
        # set the model back to training mode
        self.model.train()

    def compute_accuracy(self, logits, target):
        # compute predicted labels by taking the argmax along dimension 1 after applying softmax
        predicted_labels = torch.argmax(torch.softmax(logits, dim=1), dim=1)

        # compute accuracy
        accuracy = torch.sum(predicted_labels == target).item() / target.size(0)

        return accuracy


def pad_collate_fn(batch):
    """
    Custom collate function to handle point clouds of different shapes in the same batch.
    Strategy: batch padding (https://plainenglish.io/blog/understanding-collate-fn-in-pytorch-f9d1742647d3)

    Args:
        batch (list): List of tuples, where each tuple contains a point cloud tensor and its class index.

    Returns:
        tuple: Tuple containing a stacked point cloud tensor and a tensor of class indices.
    """
    point_clouds, class_indices = zip(*batch)

    # pad point clouds to the maximum number of points in the batch
    padded_point_clouds = pad_sequence(point_clouds, batch_first=True)
    labels = torch.tensor(class_indices)

    return padded_point_clouds, labels

def collate_fn(batch):
    """
    Custom collate function for PyTorch DataLoader.

    Args:
        batch (list): A list of samples, where each sample is a tuple (input, target).

    Returns:
        tuple: A tuple containing two tensors - stacked inputs and a tensor of targets.

    Example:
        For a batch [(input1, target1), (input2, target2), ...], collate_fn will return:
        (torch.stack([input1, input2, ...]), torch.tensor([target1, target2, ...]))
    """
    # unpack the batch into separate lists of inputs and targets
    inputs, targets = zip(*batch)

    # stack the inputs along a new dimension to form a batch tensor
    stacked_inputs = torch.stack(inputs)

    # convert the list of targets to a tensor
    target_tensor = torch.tensor(targets)

    # return a tuple containing stacked inputs and target tensor
    return stacked_inputs, target_tensor

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
    # create a new 3D figure
    fig = plt.figure()
    
    # add a 3D subplot to the figure
    ax = fig.add_subplot(111, projection="3d")
    
    # extract X, Y, Z coordinates from the point cloud
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    # scatter plot of the points with red color and spherical markers
    ax.scatter(x, y, z, c='r', marker='o')

    # set labels for the axes
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # show the plot
    plt.show()


# check if the script is being run as the main program
if __name__ == "__main__":

    # set the batch size for training and testing
    batch_size = 2

    # set the number of training epochs
    num_epochs = 100
    
    # set the base path for the dataset
    base_path = "./point-clouds/data/raw/"
    
    # determine the device for training (use GPU if available, otherwise use CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # define data transformations for training and testing
    train_transform = transforms.Compose([RandomJitterTransform(),
                                          RandomJitterTransform(),
                                          ScaleTransform()])

    test_transform = transforms.Compose([ScaleTransform()])

    # create instances of the custom dataset for training and testing
    train_dataset = CustomModelNetDataset(data_root=base_path, 
                                          transform=train_transform, train=True)    
    test_dataset = CustomModelNetDataset(data_root=base_path, 
                                         transform=test_transform, train=False)

    if batch_size > 1:
        loader_collate_arg = {"collate_fn": pad_collate_fn}
    else:
        loader_collate_arg = {"collate_fn": collate_fn}
    
    # create DataLoader instances for training and testing
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, **loader_collate_arg)    
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=4, **loader_collate_arg)

    # # visualize samples        
    # for i, (x, y) in enumerate(train_loader):
    #     print(y[0])
    #     visualize_pointcloud(x[0].squeeze())

    # create an instance of the PointNet model and move it to the specified device
    point_net = PointNet().to(device)
    
    # define the optimizer and loss function for training the model
    optimizer = torch.optim.Adam(params=point_net.parameters(), 
                                 lr=0.001, betas=(0.9, 0.999))
    loss_fn = nn.CrossEntropyLoss()

    # create an instance of the Solver class for training and validation
    solver = Solver(epochs=num_epochs, 
                    trainloader=train_loader,
                    testloader=test_loader,
                    device=device,
                    model=point_net,
                    optimizer=optimizer,
                    criterion=loss_fn,
                    patience=5)
    
    # train the neural network
    solver.train_net()
