"""
Credits: 
    https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md
"""
import os
import json
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from torchvision import models
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torchvision.models import ResNet18_Weights
from torch.utils.data import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from solver import Solver
from custom_dataset import CleanDirtyRoadDataset

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# parameters
RESIZE = 224


def reset_weights(m):
    """
    Resets the trainable parameters of the given PyTorch model.

    Args:
        m (torch.nn.Module): The PyTorch model.
    """
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            print(f"Reset trainable parameters of layer = {layer}")
            layer.reset_parameters()


def kfold_main(args):
    """
    Perform k-fold cross-validation for training a neural network.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        None
    """
    base_img_pth = Path("./data/Images")
    metadata_path = Path("./data/metadata.csv")

    # configuration options
    k_folds = 5

    # to store fold results
    results = {}

    # prepare dataset by concatenating Train/Test part; we split later.
    df = pd.read_csv(filepath_or_buffer=metadata_path)

    # define data transformations for training and testing
    if args.apply_transformations:
        train_transform = transforms.Compose([
            transforms.Resize(size=(RESIZE, RESIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
            transforms.Resize(size=(RESIZE, RESIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        train_transform = None
        test_transform = None

    # create instances of the custom dataset for training and testing
    dataset = CleanDirtyRoadDataset(X=df["filename"], y=df["label"],
                                    resize=RESIZE, data_root=base_img_pth, transform=None)

    # pin memory to speed up the host (CPU) to device (GPU) transfer
    pin = True if torch.cuda.is_available() else False

    # determine the device for training (use GPU if available, otherwise use CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice: ", device, "\n")

    # define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=SEED)

    # start print
    print("--------------------------------")

    # k-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # tensorboard specifications
        date = "_" + datetime.now().strftime("%d%m%Y-%H%M%S")
        writer = SummaryWriter("./runs/" + args.run_name + f"-fold-{fold}" + date)

        # print
        print(f"FOLD {fold}")
        print("--------------------------------")

        # # Sample elements randomly from a given list of ids, no replacement.
        # train_subsampler = SubsetRandomSampler(train_ids)
        # test_subsampler = SubsetRandomSampler(test_ids)

        # # Define data loaders for training and testing data in this fold
        # train_loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=pin,
        #                           num_workers=os.cpu_count(), sampler=train_subsampler)
        # test_loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=pin, 
        #                          num_workers=os.cpu_count(), sampler=test_subsampler)

        # define data loaders for training and testing data in this fold
        train_dataset_fold = CleanDirtyRoadDataset(X=df["filename"].iloc[train_ids],
                                                   y=df["label"].iloc[train_ids],
                                                   RESIZE=RESIZE, data_root=base_img_pth,
                                                   transform=train_transform)
        test_dataset_fold = CleanDirtyRoadDataset(X=df["filename"].iloc[test_ids],
                                                  y=df["label"].iloc[test_ids],
                                                  RESIZE=RESIZE, data_root=base_img_pth,
                                                  transform=test_transform)

        train_loader = DataLoader(train_dataset_fold, batch_size=args.batch_size, pin_memory=pin,
                                  num_workers=os.cpu_count(), shuffle=True)
        test_loader = DataLoader(test_dataset_fold, batch_size=args.batch_size, pin_memory=pin,
                                 num_workers=os.cpu_count(), shuffle=False)

        # initialize the neural network
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # freeze all parameters for fine-tuning (feature learning)
        for param in model.parameters():
            param.requires_grad = False

        # add a new final layer
        nr_filters = model.fc.in_features
        model.fc = nn.Linear(nr_filters, args.num_classes)

        # reset the new layer's parameters to avoid weight leaks
        ###########################
        model.fc.reset_parameters()
        ###########################
        
        model = model.to(device)

        # # to avoid weight leaks
        # ###########################
        # model.apply(reset_weights)
        # ###########################

        # define the optimizer and loss function for training the model
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999))
        # optimizer = torch.optim.RMSprop(params=model.parameters(), lr=0.004014796)
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

        # evaluate the model on the test set
        model_results = solver.eval_model()
        model_results["test_name"] = args.run_name

        # print accuracy
        print(f"\nAccuracy for fold {fold}: {model_results['model_acc'] * 100:.3f}%")
        print("\n--------------------------------")
        results[fold] = model_results  # dictionary of (dictionary) results

        # clean up memory
        del model, optimizer, loss_fn
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(f"./statistics/{args.run_name}-kfold-model_results.json", "w") as outfile:
        json.dump(results, outfile)

    # print fold results
    print(f"K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS")
    print("--------------------------------")
    sum_acc = 0.0
    sum_loss = 0.0

    for i, fold_result in results.items():
        acc = fold_result["model_acc"]
        loss = fold_result["model_loss"]
        print(f"Fold {i}: Accuracy = {acc * 100:.3f}%, Loss = {loss}")
        sum_acc += acc
        sum_loss += loss

    avg_acc = sum_acc / len(results)
    avg_loss = sum_loss / len(results)

    print(f"\nAverage Accuracy: {avg_acc * 100:.3f}%")
    print(f"Average Loss: {avg_loss:.3f}\n")
