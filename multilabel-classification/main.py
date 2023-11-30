import os
import torch
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from sklearn.utils import shuffle
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from model import ResNet18
from model import MultiLabelImageClassifier

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


class CustomImageDataset(Dataset):
    def __init__(self, dataframe, skipcols, data_root, transform, resize):
        """
        Custom dataset for loading 2D images.
        https://www.kaggle.com/datasets/meherunnesashraboni/multi-label-image-classification-dataset

        Args:
            data_root (str): Root directory of the dataset.
            transform (callable): Optional transform to be applied on a sample.
            train (bool): Flag indicating whether to load training or testing data.
        """
        super(CustomImageDataset, self).__init__()
        self.df = dataframe
        self.skipcols = skipcols
        self.data_root = data_root
        self.transform = transform
        self.resize = resize

    def __getitem__(self, index):
        """
        Retrieves a specific item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: Tuple containing the 3D point cloud data and its corresponding class index.
        """
        img_name = self.df.iloc[index]["Image_Name"]
        img_labels = self.df.iloc[index, self.skipcols:]
        img_pth = self.data_root / img_name
        # RGB prevent from grayscale images in the dataset
        img = Image.open(img_pth).convert("RGB")
        labels = torch.tensor(img_labels,
                              dtype=torch.float32)
        
        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            transform = transforms.Compose([transforms.Resize(size=(self.resize, self.resize)),
                                            transforms.ToTensor()])   
            img_tensor = transform(img)
        
        return img_tensor, labels

    def __len__(self):
        """
        Returns the total number of items in the dataset.

        Returns:
            int: Total number of items in the dataset.
        """
        return len(self.df)


class EarlyStopping(object):
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Copyright (c) 2018 Bjarte Mehus Sunde
    """
    def __init__(self, patience=5, delta=0, path="./multilabel-classification/checkpoints/checkpoint.pt", verbose=False):
        """
        Args:
            patience (int): How long to wait after the last time validation loss improved.
                            Default: 5
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        # initialize the parameters
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose

        # counter to track the number of epochs without improvement
        self.counter = 0

        # best validation score initialized to None
        self.best_score = None

        # flag to indicate if early stopping criteria are met
        self.early_stop = False

        # minimum validation loss initialized to positive infinity
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        # check if it's the first validation, or there is an improvement, or no improvement
        if self.best_score is None:
            self._handle_first_validation(val_loss, model)
        elif val_loss > self.best_score + self.delta:
            self._handle_no_improvement(val_loss, model)
        else:
            self._handle_improvement(val_loss, model)

    def _handle_first_validation(self, val_loss, model):
        # handle the case of the first validation
        self.best_score = val_loss
        self.save_checkpoint(val_loss, model)

    def _handle_no_improvement(self, val_loss, model):
        # handle the case of no improvement in validation loss
        self.counter += 1
        print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
        if self.counter >= self.patience:
            # if the counter exceeds patience, set the early_stop flag
            self.early_stop = True

    def _handle_improvement(self, val_loss, model):
        # handle the case of an improvement in validation loss
        self.counter = 0
        self.best_score = val_loss
        self.save_checkpoint(val_loss, model)

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        # save the model's state_dict if there is a decrease in validation loss
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...")
        torch.save(obj=model.state_dict(), f=self.path)
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

            predictions = torch.tensor([])
            targets = torch.tensor([])

            # use tqdm for a progress bar during training
            loop = tqdm(iterable=enumerate(self.trainloader),
                        total=len(self.trainloader),
                        leave=True)

            for _, (x_train, y_train) in loop:
                # move data and labels to the specified device
                x_train = x_train.to(self.device)
                y_train = y_train.to(self.device)

                # forward pass: compute predicted outputs by passing inputs to the model
                logits = self.model(x_train)

                # calculate the loss
                loss = self.criterion(logits, y_train)

                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()

                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                # perform a single optimization step (parameter update)
                self.optimizer.step()

                # record training loss
                train_losses.append(loss.item())

                # since we are using BCEWithLogitsLoss
                # logits --> probabilities --> labels
                probs = torch.sigmoid(logits)
                y_pred = torch.round(probs)

                # record predictions and true labels
                predictions = torch.cat([predictions, y_pred], dim=0)
                targets = torch.cat([targets, y_train], dim=0)

            self.compute_metrics(predictions=predictions,
                                 targets=targets,
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

        predictions = torch.tensor([])
        targets = torch.tensor([])

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
                logits = self.model(x_valid)

                # calculate the loss
                loss = self.criterion(logits, y_valid)

                # record validation loss
                valid_losses.append(loss.item())

                # since we are using BCEWithLogitsLoss
                # logits --> probabilities --> labels
                probs = torch.sigmoid(logits)
                y_pred = torch.round(probs)

                # record predictions and true labels
                predictions = torch.cat([predictions, y_pred], dim=0)
                targets = torch.cat([targets, y_valid], dim=0)

            self.compute_metrics(predictions=predictions,
                                 targets=targets,
                                 train=False)
            
        # set the model back to training mode
        self.model.train()

    def compute_metrics(self, predictions, targets, train):
        # compute accuracy
        # accuracy = torch.sum(predictions == targets).item() / (targets.size(0) * targets.size(1))

        true_positives = torch.sum((predictions == 1) & (targets == 1)).item()
        true_negatives = torch.sum((predictions == 0) & (targets == 0)).item()
        false_positives = torch.sum((predictions == 1) & (targets == 0)).item()
        false_negatives = torch.sum((predictions == 0) & (targets == 1)).item()

        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives 
                                                        + false_positives + false_negatives + 1e-20)
        precision = true_positives / (true_positives + false_positives + 1e-20)
        recall = true_positives / (true_positives + false_negatives + 1e-20)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-20)


        # print accuracy based on training or validation
        accuracy_type = "Train" if train else "Valid"
        print(f"\n{accuracy_type} accuracy: {accuracy:.3f} | "
              f"precision: {precision:.3f} | recall: {recall:.3f} | f1_score: {f1_score:.3f} ")


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

    parser.add_argument("--batch_size", type=int, default=16,
                        help="the batch size for training and validation data")

    # https://nvlabs.github.io/eccv2020-mixed-precision-tutorial/files/szymon_migacz-pytorch-performance-tuning-guide.pdf    
    parser.add_argument("--workers", type=int, default=4,
                        help="the number of workers in the data loader")
    #######################################################################################

    # training-parameters (2)
    #######################################################################################
    parser.add_argument("--lr", type=float, default=0.001,
                        help="the learning rate for optimization")

    parser.add_argument("--loss", type=str, default="bcewll",
                        choices=["bcewll"],
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

    parser.add_argument("--checkpoint_path", type=str, default="./multilabel-classification/checkpoints", 
                        help="the path to save the trained model")

    parser.add_argument("--num_classes", type=int, default=16,
                        help="the number of classes to predict with the final Linear layer")
    #######################################################################################

    # data-path
    #######################################################################################
    parser.add_argument("--raw_data_path", type=str, default="./multilabel-classification/data/apparel-images-dataset/",
                        help="path where to get the raw-dataset")
    #######################################################################################

    # data transformation
    #######################################################################################
    parser.add_argument("--apply_transformations", action="store_true", # default=True,
                        help="indicates whether to apply transformations to images")
    #######################################################################################

    return parser.parse_args()

# check if the script is being run as the main program
if __name__ == "__main__":
    # parse command line arguments
    args = get_args()

    # if the checkpoint folder doesn't exist, create it
    if not os.path.isdir(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    
    # determine the device for training (use GPU if available, otherwise use CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # read the CSV file containing data for multi-label classification
    df = pd.read_csv(filepath_or_buffer="./multilabel-classification/data/multilabel_classification(2).csv")

    # specify the base image path
    base_img_pth = Path("./multilabel-classification/data/images")

    # filter the DataFrame to include only rows with existing image files
    filtered_df = df[[os.path.isfile(base_img_pth / img_pth) for img_pth in df["Image_Name"]]]

    # shuffle the dataset
    filtered_df = shuffle(filtered_df)
    
    # split the dataset into training and testing sets
    train_df, test_df = train_test_split(filtered_df, test_size=0.2)

    # set skipcols of the dataframe
    skipcols = 2

    # set resize dimensions for image preprocessing
    resize = 224

    # define data transformations for training and testing
    if args.apply_transformations:
        train_transform = transforms.Compose([transforms.Resize(size=(resize, resize)),
                                              transforms.RandomHorizontalFlip(p=0.5),
                                              transforms.RandomVerticalFlip(p=0.5),
                                              transforms.RandomRotation(degrees=15),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], 
                                                                   [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize(size=(resize, resize)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])
    else:
        train_transform = None
        test_transform = None

    # create instances of the custom dataset for training and testing
    train_dataset = CustomImageDataset(dataframe=train_df, skipcols=skipcols, data_root=base_img_pth, 
                                       transform=train_transform, resize=resize)    
    test_dataset = CustomImageDataset(dataframe=test_df, skipcols=skipcols, data_root=base_img_pth, 
                                      transform=test_transform, resize=resize)
        
    # create DataLoader instances for training and testing
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers)    
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.workers)

    # create an instance of the ResNet18 model and move it to the specified device
    x, _ = next(iter(train_loader))
    net = ResNet18(input_size=x[0].shape, output_size=args.num_classes).to(device)
    # net = MultiLabelImageClassifier(num_classes=args.num_classes).to(device)
    
    # define the optimizer and loss function for training the model
    optimizer = torch.optim.Adam(params=net.parameters(), 
                                 lr=args.lr, betas=(0.9, 0.999))
    loss_fn = nn.BCEWithLogitsLoss()

    # create an instance of the Solver class for training and validation
    solver = Solver(epochs=args.num_epochs, 
                    trainloader=train_loader,
                    testloader=test_loader,
                    device=device,
                    model=net,
                    optimizer=optimizer,
                    criterion=loss_fn,
                    patience=args.patience)
    
    # train the neural network
    solver.train_net()
