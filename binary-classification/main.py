import os
import torch
import random
import json
import argparse
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from torchvision import  models
from torchvision import transforms
import pytorch_model_summary as pms
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

from solver import Solver
from models import BinaryImageClassifier
from custom_dataset import CleanDirtyRoadDataset


# reproducibility
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True


def get_args():
    parser = argparse.ArgumentParser()

    # model-infos
    #######################################################################################
    parser.add_argument("--run_name", type=str, default="test_default",
                        help="the name assigned to the current run")

    parser.add_argument("--model_name", type=str, default="binaryclassif_img_model",
                        help="the name of the model to be saved or loaded")
    #######################################################################################

    # training-parameters (1)
    #######################################################################################
    parser.add_argument("--num_epochs", type=int, default=200,
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

    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints", 
                        help="the path to save the trained model")

    parser.add_argument("--num_classes", type=int, default=1,
                        help="the number of classes to predict with the final Linear layer")
    #######################################################################################

    # data-path
    #######################################################################################
    parser.add_argument("--raw_data_path", type=str, default="./data/Images",
                        help="path where to get the raw-dataset")
    #######################################################################################

    # data transformation
    #######################################################################################
    parser.add_argument("--apply_transformations", action="store_true", # default=True,
                        help="indicates whether to apply transformations to images")
    #######################################################################################

    # # Google Colab
    # #######################################################################################
    # parser.add_argument("-f", "--file", required=False)
    # #######################################################################################

    return parser.parse_args()

def main(args):
    # tensorboard specifications
    date =  "_" + datetime.now().strftime("%d%m%Y-%H%M%S")
    writer = SummaryWriter("./runs/" + args.run_name + date)
 
    base_img_pth = Path("./data/Images")
    metadata_path = Path("./data/metadata.csv")

    df = pd.read_csv(filepath_or_buffer=metadata_path)

    X_train, X_test, y_train, y_test = train_test_split(df["filename"], df["label"],
                                                        test_size=0.2, random_state=seed, shuffle=True)
    
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
    train_dataset = CleanDirtyRoadDataset(X=X_train, y=y_train, resize=resize, 
                                          data_root=base_img_pth, transform=train_transform)
    test_dataset = CleanDirtyRoadDataset(X=X_test, y=y_test, resize=resize,
                                         data_root=base_img_pth, transform=test_transform)
    
    # pin_memory: speed up the host (cpu) to device (gpu) transfer
    pin = True if torch.cuda.is_available() else False

    # create DataLoader instances for training and testing
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=os.cpu_count(), pin_memory=pin)
    test_loader = DataLoader(dataset=test_dataset,batch_size=args.batch_size, 
                             shuffle=False, num_workers=os.cpu_count(), pin_memory=pin)
    
    # determine the device for training (use GPU if available, otherwise use CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice: ", device)
    
    finetuning = True
    
    if finetuning:
        # model = models.resnet18(pretrained=True)
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # freeze all params
        for param in model.parameters():
            param.requires_grad = False
        
        # add a new final layer
        # number of input features of last layer
        nr_filters = model.fc.in_features
        model.fc = nn.Linear(nr_filters, args.num_classes)
        model = model.to(device)
    # else:
    #     # create an instance of the model and move it to the specified device
    #     model = BinaryImageClassifier(output_size=args.num_classes).to(device)

    images, _ = next(iter(train_loader))
    images = images.to(device)
    img_grid = torchvision.utils.make_grid(images)
    writer.add_image("sample-images", img_grid, global_step=0)

    print("\nModel: ")
    pms.summary(model, torch.zeros(images.shape, device=device), max_depth=5, print_summary=True)

    # define the optimizer and loss function for training the model
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=args.lr, betas=(0.9, 0.999))
    loss_fn = nn.BCEWithLogitsLoss()

    # create an instance of the Solver class for training and validation
    solver = Solver(epochs=args.num_epochs, 
                    writer=writer,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    device=device,
                    model=model,
                    optimizer=optimizer,
                    criterion=loss_fn,
                    patience=args.patience)
    
    # train the neural network
    solver.train_net()

    model_reslts = solver.eval_model()
    model_reslts["test_name"] = args.run_name
    with open(f"./statistics/{args.run_name}-model_reslts.json", "w") as outfile:
        json.dump(model_reslts, outfile)


# check if the script is being run as the main program
if __name__ == "__main__":
    # parse command line arguments
    args = get_args()
    
    # if folder doesn't exist, then create it
    if not os.path.isdir(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.isdir("statistics"):
        os.makedirs("statistics")
    
    print(f"\n{args}")
    main(args)
