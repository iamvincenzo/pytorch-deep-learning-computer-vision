import os
import torch
import pathlib
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from model import Model


class CustomImageDataset(Dataset):
    def __init__(self, data_root, transform, train):
        """
        Custom dataset for loading 2D images.

        Args:
            data_root (str): Root directory of the dataset.
            transform (callable): Optional transform to be applied on a sample.
            train (bool): Flag indicating whether to load training or testing data.
        """
        super(CustomImageDataset, self).__init__()
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

    def load_image(self, file_path):
        """
        Loads 2D images.

        Args:
            file_path (str): Path to the OFF file.

        Returns:
            torch.Tensor: 2D image as a PyTorch tensor.
        """
        pass

    def __getitem__(self, index):
        """
        Retrieves a specific item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: Tuple containing the 3D point cloud data and its corresponding class index.
        """
        image_path = self.file_paths[index]

        class_name = image_path.parent.parent.name

        image = self.load_point_cloud(file_path=image_path)

        if self.transform is not None:
            image = self.transform(image)
        
        return image, self.class_to_idx[class_name]

    def __len__(self):
        """
        Returns the total number of items in the dataset.

        Returns:
            int: Total number of items in the dataset.
        """
        return len(self.file_paths)


class EarlyStopping:
    """ 
    Early stops the training if validation loss doesn't improve after a given patience.
    Copyright (c) 2018 Bjarte Mehus Sunde
    """
    def __init__(self, patience=7, verbose=False, delta=0, path="./point-clouds/checkpoints/checkpoint.pt", trace_func=print):
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
                y_pred, feature_t = self.model(x_train)

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
            self.compute_accuracy(logits=all_preds, 
                                  target=all_targets,
                                  train=True)

            # validate the model on the validation set
            self.valid_net(valid_losses=valid_losses)

            # print training/validation statistics
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            # print some statistics
            print(f"\nEpoch[{epoch + 1}/{self.epochs}] | train-loss: {train_loss:.4f} | "
                  f"validation-loss: {valid_loss:.4f}")

            # clear lists to track next epoch
            train_losses = []
            valid_losses = []

            # early stopping checks for improvement in validation loss
            early_stopping(valid_loss, self.model)

            if early_stopping.early_stop:
                print("Early stopping...")
                break
        
        print("\nTraining model Done...\n")

    def valid_net(self, valid_losses):
        """
        Validates the neural network on the specified DataLoader for validation data.

        Records validation losses in the provided list.

        Args:
            valid_losses (list): List to record validation losses.
        """
        print(f"\nStarting validation...\n")

        predictions_list = []
        targets_list = []

        self.model.eval()

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
                y_pred, feature_t = self.model(x_valid)

                # calculate the loss
                loss = self.criterion(y_pred, y_valid)

                # record validation loss
                valid_losses.append(loss.item())

                # record predictions and true labels
                predictions_list.append(y_pred)
                targets_list.append(y_valid)

            all_preds = torch.cat(predictions_list, dim=0)
            all_targets = torch.cat(targets_list, dim=0)
            self.compute_accuracy(logits=all_preds,
                                  target=all_targets,
                                  train=False)
            
        # set the model back to training mode
        self.model.train()

    def compute_accuracy(self, logits, target, train):
        # compute predicted labels by taking the argmax along dimension 1 after applying softmax
        predicted_labels = torch.argmax(torch.softmax(logits, dim=1), dim=1)

        # compute accuracy
        accuracy = torch.sum(predicted_labels == target).item() / target.size(0)

        # print accuracy based on training or validation
        accuracy_type = "Train" if train else "Valid"
        print(f"\n{accuracy_type} accuracy: {accuracy:.4f}")


def get_args():
    parser = argparse.ArgumentParser()

    # model-infos
    #######################################################################################
    parser.add_argument("--run_name", type=str, default="test_1",
                        help="the name assigned to the current run")

    parser.add_argument("--model_name", type=str, default="pointnet",
                        help="the name of the model to be saved or loaded")
    #######################################################################################

    # training-parameters (1)
    #######################################################################################
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="the total number of training epochs")

    parser.add_argument("--batch_size", type=int, default=2,
                        help="the batch size for training and validation data")

    # https://nvlabs.github.io/eccv2020-mixed-precision-tutorial/files/szymon_migacz-pytorch-performance-tuning-guide.pdf    
    parser.add_argument("--workers", type=int, default=4,
                        help="the number of workers in the data loader")
    #######################################################################################

    # training-parameters (2)
    #######################################################################################
    parser.add_argument("--lr", type=float, default=0.001,
                        help="the learning rate for optimization")

    parser.add_argument("--loss", type=str, default="cel",
                        choices=["cel"],
                        help="the loss function used for model optimization")

    parser.add_argument("--opt", type=str, default="Adam", 
                        choices=["SGD", "Adam"],
                        help="the optimizer used for training")

    parser.add_argument("--patience", type=int, default=5,
                        help="the threshold for early stopping during training")
    #######################################################################################

    # training-parameters (3)
    #######################################################################################
    parser.add_argument("--load_model", action="store_true",
                        help="determines whether to load the model from a checkpoint")

    parser.add_argument("--checkpoint_path", type=str, default="./point-clouds/checkpoints", 
                        help="the path to save the trained model")

    parser.add_argument("--num_classes", type=int, default=10,
                        help="the number of classes to predict with the final Linear layer")
    #######################################################################################

    # data-path
    #######################################################################################
    parser.add_argument("--raw_data_path", type=str, default="./point-clouds/data/raw/",
                        help="path where to get the raw-dataset")
    #######################################################################################

    # data transformation
    #######################################################################################
    parser.add_argument("--apply_transformations", action="store_true",
                        help="indicates whether to apply transformations to images")
    #######################################################################################

    return parser.parse_args()

# check if the script is being run as the main program
if __name__ == "__main__":
    args = get_args()

    # if folder doesn't exist, then create it
    if not os.path.isdir(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    
    # determine the device for training (use GPU if available, otherwise use CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # define data transformations for training and testing
    if args.apply_transformations:
        train_transform = transforms.Compose([])
        test_transform = transforms.Compose([])
    else:
        train_transform = None
        test_transform = None

    # create instances of the custom dataset for training and testing
    train_dataset = CustomImageDataset(data_root=args.raw_data_path,
                                       transform=train_transform, train=True)    
    test_dataset = CustomImageDataset(data_root=args.raw_data_path,
                                      transform=test_transform, train=False)
    
    # create DataLoader instances for training and testing
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers)    
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.workers)

    # # visualize samples        
    # for _, batch in enumerate(train_loader):
    #     for x, y in zip(*batch):
    #         key = [key for key, value 
    #                in train_dataset.class_to_idx.items() 
    #                if value == y.item()]
    #         visualize_image(x.squeeze(), key[0])

    # create an instance of the PointNet model and move it to the specified device
    point_net = Model(num_classes=args.num_classes).to(device)
    
    # define the optimizer and loss function for training the model
    optimizer = torch.optim.Adam(params=point_net.parameters(), 
                                 lr=args.lr, betas=(0.9, 0.999))
    loss_fn = nn.BCEWithLogitsLoss()

    # create an instance of the Solver class for training and validation
    solver = Solver(epochs=args.num_epochs, 
                    trainloader=train_loader,
                    testloader=test_loader,
                    device=device,
                    model=point_net,
                    optimizer=optimizer,
                    criterion=loss_fn,
                    patience=args.patience)
    
    # train the neural network
    solver.train_net()
