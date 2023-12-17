import os
import torch
import argparse
import pandas as pd
import torch.nn as nn
from pathlib import Path
from sklearn.utils import shuffle
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from solver import Solver
from models import ResNet18
from dataset import CustomImageDataset
from models import MultiLabelImageClassifier

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


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
    parser.add_argument("--apply_transformations", action="store_true", default=True,
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
