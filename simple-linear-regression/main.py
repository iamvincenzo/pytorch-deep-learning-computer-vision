import copy
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from models import LinearRegrNet
from models import RawLinearRegrNet


class CustomDataset(Dataset):
    """
    A custom PyTorch dataset.

    Parameters:
        X (list or numpy array): Input data.
        y (list or numpy array): Target data.

    Attributes:
        X (list or numpy array): Input data.
        y (list or numpy array): Target data.

    Methods:
        __getitem__(idx): Gets the data sample at the specified index.
        __len__(): Returns the total number of samples in the dataset.
    """
    def __init__(self, X, y):
        """
        Initializes the CustomDataset.

        Args:
            X (list or numpy array): Input data.
            y (list or numpy array): Target data.
        """
        super().__init__()

        if len(X) != len(y):
            raise ValueError("Input and target data must have the same length.")

        self.X = X
        self.y = y

    def __getitem__(self, index):
        """
        Gets the data sample at the specified index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the input data and target data.
        """
        return self.X[index], self.y[index]

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.X)


class Solver(object):
    """
    A class for training and evaluating PyTorch models.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        loss_fn (torch.nn.Module): The loss function for training.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        test_loader (torch.utils.data.DataLoader): DataLoader for testing data.
        device (torch.device): The device on which the model will be trained and evaluated.
        save_path (str): The file path where the trained model will be saved.
        epochs (int): Number of training epochs.

    Methods:
        train_model(): Train the PyTorch model and print training and testing losses.
        evaluate_model(): Evaluate the PyTorch model on the test set and return predictions and average loss.
        save_model(): Save the state dictionary of the PyTorch model to a specified file path.
        load_model(): Load the state dictionary of the PyTorch model from a specified file path.
    """
    def __init__(self, model, loss_fn, optimizer, train_loader, test_loader, device, save_path, epochs):
        """
        Initializes the Solver with the necessary components.

        Args:
            model (torch.nn.Module): The PyTorch model to be trained.
            loss_fn (torch.nn.Module): The loss function for training.
            optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            test_loader (torch.utils.data.DataLoader): DataLoader for testing data.
            device (torch.device): The device on which the model will be trained and evaluated.
            save_path (str): The file path where the trained model will be saved.
            epochs (int): Number of training epochs.
        """
        super().__init__()
        self.model = model
        self.criterion = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.save_path = save_path
        self.epochs = epochs

    def train_model(self):
        """
        Train the PyTorch model and print training and testing losses.

        Returns:
            torch.Tensor: Predictions on the test set.
        """
        print(f"\nStarting training process...\n")

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0

            loop = tqdm(iterable=self.train_loader, total=len(self.train_loader), leave=True)

            for _, (X_train, y_train) in enumerate(loop):
                X_train = X_train.to(device=self.device)
                y_train = y_train.to(device=self.device)
                
                # 1. forward pass
                y_pred = self.model(X_train)
                
                # 2. calculate the loss
                loss = self.criterion(y_pred, y_train)

                # 3. optimizer zero grad
                self.optimizer.zero_grad()

                # 4. perform backpropagation
                loss.backward()

                # 5. perform gradient descent
                self.optimizer.step()

                total_loss += loss.item()

            # calculate average training loss for the epoch
            average_loss = total_loss / len(self.train_loader)

            # 6. evaluation on the test set
            y_pred_test, loss_test = self.evaluate_model()
                
            # print loss for every epoch
            print(f"\nEpoch [{epoch}/{self.epochs}], "
                  f"Train Loss: {average_loss:.4f}, Test Loss: {loss_test:.4f}\n")
            
        # save the trained model
        self.save_model()
                
        return y_pred_test

    def evaluate_model(self):
        """
        Evaluate the PyTorch model on the test set and return predictions and average loss.

        Returns:
            torch.Tensor: Predictions on the test set.
            float: The average loss on the test set.
        """
        predictions = torch.tensor([])
        total_loss = 0.0

        self.model.eval()

        with torch.inference_mode():
            loop = tqdm(iterable=self.test_loader, total=len(self.test_loader), leave=True)

            for _, (X_test, y_test) in enumerate(loop):
                X_test = X_test.to(device=self.device)
                y_test = y_test.to(device=self.device)

                y_pred_test = self.model(X_test)
                loss_test = self.criterion(y_pred_test, y_test)
                total_loss += loss_test.item()
                predictions = torch.cat([predictions, y_pred_test])

        # calculate average loss on the test set
        average_loss = total_loss / len(self.test_loader)

        self.model.train()

        return predictions, average_loss

    def save_model(self):
        """
        Save the state dictionary of the PyTorch model to a specified file path.

        Returns:
            None
        """
        torch.save(obj=self.model.state_dict(), f=self.save_path)

    def load_model(self):
        """
        Load the state dictionary of the PyTorch model from a specified file path.

        Returns:
            None
        """
        self.model.load_state_dict(torch.load(f=self.save_path, map_location=self.device))


def plot_data(X_train, y_train, X_test, y_test, predictions=None):
    """
    Plot the training and testing data.

    Args:
        X_train (torch.Tensor): Input training data.
        y_train (torch.Tensor): Target training data.
        X_test (torch.Tensor): Input testing data.
        y_test (torch.Tensor): Target testing data.
        predictions (torch.Tensor, optional): Predicted data. Defaults to None.
    """
    # create a figure with a specified size
    plt.figure(figsize=(18, 5))

    # plot the training data in blue
    plt.scatter(X_train, y_train, c="b", label="Train data")

    # plot the testing data in green
    plt.scatter(X_test, y_test, c="g", label="Test data")

    # plot the predicted data in red if predictions are provided
    if predictions is not None:
        plt.scatter(X_test, predictions, c="r", label="Predicted Data")

    # set labels and title
    plt.xlabel("X-axis Value")
    plt.ylabel("Y-axis Label")
    plt.title("Data Plot")

    # aAdd a legend to the plot
    plt.legend()

    # show the plot
    plt.show()


if __name__ == "__main__":
    # reproducibility
    torch.manual_seed(42)
    
    # check if GPU is available, else use CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # create a Linear Regression model
    net = LinearRegrNet().to(device=device)
    # net = RawLinearRegrNet().to(device=device)

    # generate synthetic data
    X = torch.arange(0, 1, 0.02)
    w, b = 0.7, 0.3
    y = w * X + b

    # split data into train and test sets
    train_split = int((0.8 * len(X)))
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

    # add a singleton dimension to tensors
    X_train, y_train = X_train.unsqueeze(1), y_train.unsqueeze(1)
    X_test, y_test = X_test.unsqueeze(1), y_test.unsqueeze(1)

    batch_size = 40
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # define the L1 loss function
    loss_fn = nn.L1Loss()

    # define the SGD optimizer
    learning_rate = 0.01
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    # save the model
    MODEL_PATH = Path("./simple-linear-regression/checkpoints")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_NAME = "01_pytorch_model.pt"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    param_before = copy.deepcopy(net.state_dict())

    solver = Solver(model=net, 
                    loss_fn=loss_fn, 
                    optimizer=optimizer, 
                    train_loader=train_loader,
                    test_loader=test_loader, 
                    device=device, 
                    save_path=MODEL_SAVE_PATH, 
                    epochs=200)
    
    # train the model and get predictions on the test set
    y_pred_test = solver.train_model()

    print(f"\nParameters before training: {param_before}")
    print(f"Parameters after training: {net.state_dict()}")
    print(f"True parameters: (weight, {w}) and (bias, {b})")

    # plot the data and predictions
    plot_data(X_train, y_train, X_test, y_test, y_pred_test)

    # load a model
    solver.load_model()
    print(f"\nParameters of loaded model: {solver.model.state_dict()}\n")
